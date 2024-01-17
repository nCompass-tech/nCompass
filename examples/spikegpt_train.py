import ncompass.loaders.train as train_loaders

from ncompass.models import download_from_hf
from ncompass.models.snn.spike_gpt.train import SpikeGPT
from ncompass.models.snn.spike_gpt import SpikeGPTConfig, get_tokenizer

from ncompass.internal.third_party.spikingjelly.clock_driven import functional

from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import get_scheduler

from tqdm import tqdm
from pathlib import Path

import os
import math
import torch

def setup_config():
    home_dir = os.getenv("HOME")
    tokenizer_json_path = home_dir + "/nCompass/ncompass/models/snn/spike_gpt/20B_tokenizer.json"
    model_path = download_from_hf(repo_id = "ridger/SpikeGPT-OpenWebText-216M"\
                                  , filename = "SpikeGPT-216M.pth")
    config = SpikeGPTConfig(token_mode = "pile"
                            , word_name = [tokenizer_json_path, tokenizer_json_path]
                            , unknown_char = None
                            , vocab_size = 50277
                            , hidden_size = 768
                            , num_hidden_layers = 18
                            , ctx_len = 1024
                            , tokenizer_class = 'SpikeGPT'
                            , temperature = 1.5
                            , top_p = 0.7
                            , name_or_path = model_path
                            , mix_mode = 'RWKV'
                            , use_cuda = True)
    return config

def produce_dataloaders(config, train_batch_size, eval_batch_size):
    tokenizer = get_tokenizer(config)
    dataset   = train_loaders.load_dataset('wikitext',
                                           'wikitext-2-raw-v1',
                                           tokenizer=tokenizer,
                                           num_workers=4)
    val_dataloader   = train_loaders.get_dataloader(dataset['validation'],
                                                    shuffle=True,
                                                    batch_size=eval_batch_size)
    train_dataloader = train_loaders.get_dataloader(dataset['train'],
                                                    shuffle=True,
                                                    batch_size=train_batch_size)
    return train_dataloader, val_dataloader

def produce_model(config):
    model = SpikeGPT(config)
    return model

def produce_optimizer(model, learning_rate):
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

def produce_lr_scheduler(optimizer, num_warmup_steps, gradient_accumulation_steps, max_train_steps):
    return get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

def calculate_train_details(train_dataloader, gradient_accumulation_steps, num_train_epochs):
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    return num_update_steps_per_epoch, max_train_steps, num_train_epochs

def setup_checkpointing(checkpoint_dir):
    checkpoint_path = Path(checkpoint_dir).absolute()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path

def resume_from_checkpoint(checkpoint_path, start_epoch, accelerator):
    checkpoint_path = checkpoint_path.joinpath(f'epoch-{start_epoch}.pth')
    accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
    accelerator.load_state(checkpoint_path)

def train_batch_loss_averaging(avg_loss, batch_num, loss):
    if avg_loss < 0: avg_loss = loss
    else:
        factor = 1 / (batch_num + 1)
        avg_loss = avg_loss * (1.0 - factor) + loss * factor
    return avg_loss

def run_train_batch(model,
                    accelerator,
                    data,
                    optimizer,
                    lr_scheduler):
    with accelerator.accumulate(model):
        with torch.set_grad_enabled(True):
            outputs = model(**data)  # forward the model
            functional.reset_net(model)

        model.zero_grad()
        loss = outputs.loss
        # We keep track of the loss at each epoch
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    # Checks if the accelerator has performed an optimization step behind the scenes
    curr_loss = loss.item()
    
    return curr_loss
    
def run_validation_batch(model, data):
    with torch.set_grad_enabled(False):
        outputs = model(**data)
        functional.reset_net(model)
    return outputs.loss

def run_epoch(model,
              train_dataloader,
              val_dataloader,
              accelerator,
              optimizer,
              lr_scheduler,
              per_device_eval_batch_size,  
              max_train_steps,
              completed_steps,
              avg_loss,
              progress_bar):
    model.train()
    steps_completed = 0
    for batch_num, batch_data in enumerate(train_dataloader):
        loss = run_train_batch(model, accelerator, batch_data, optimizer, lr_scheduler)
        avg_loss = train_batch_loss_averaging(avg_loss, batch_num, loss)

        progress_bar.set_description(f"training steps: {batch_num}/{len(train_dataloader)}\
                                       total training steps: {completed_steps + steps_completed}\
                                       ppl {math.exp(avg_loss):.2f}\
                                       loss {avg_loss:.4f}")
        if accelerator.sync_gradients:
            progress_bar.update(1)
            steps_completed += 1

        if int(completed_steps + steps_completed) >= max_train_steps: break

    model.eval()
    losses = []
    for batch_num, batch_data in enumerate(val_dataloader):
        loss = run_validation_batch(model, batch_data)
        losses.append(accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size)))
        progress_bar.set_description(f"eval steps: {batch_num}/{len(val_dataloader)}\
                                       ppl {math.exp(avg_loss):.2f}\
                                       loss {avg_loss:.4f}")
    losses = torch.cat(losses)
    return steps_completed, losses, avg_loss

def compute_perplexity(losses):
    try: eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return eval_loss, perplexity

def run_training(start_epoch,
               num_training_epochs,
               model,
               train_dataloader,
               val_dataloader,
               accelerator,
               optimizer,
               lr_scheduler,
               per_device_eval_batch_size,
               max_train_steps,
               checkpoint_path,
               progress_bar):
    completed_steps = 0
    logfile = checkpoint_path.joinpath('progress.log')
    logger = open(logfile, 'w')
    print(f'epoch, perplexity, train_loss, eval_loss', file=logger)
    avg_loss = -1
    for epoch in range(start_epoch, num_training_epochs):
        steps_completed, losses, avg_loss = run_epoch(model,
                                                      train_dataloader,
                                                      val_dataloader,
                                                      accelerator,
                                                      optimizer,
                                                      lr_scheduler,
                                                      per_device_eval_batch_size,
                                                      max_train_steps,
                                                      completed_steps,
                                                      avg_loss,
                                                      progress_bar)
        eval_loss, perplexity = compute_perplexity(losses)

        progress_bar.set_description(f"epoch {epoch}: ppl {perplexity} eval_loss: {eval_loss}")
        print(f"{epoch}, {perplexity}, {avg_loss}, {eval_loss}", file=logger)
        
        save_path = checkpoint_path.joinpath(f'epoch-{epoch}')
        accelerator.save_state(save_path)
        accelerator.save_model(model, save_path, safe_serialization=False)
        completed_steps += steps_completed
    logger.close()

def main():
    # to be made into training config
    num_warmup_steps = 0
    gradient_accumulation_steps = 1
    per_device_train_batch_size = 3
    per_device_eval_batch_size = 8
    learning_rate = 6e-4
    start_epoch = 0
    num_train_epochs = 10
    checkpoint_dir = '.checkpoint'

    set_seed(10)
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

    config = setup_config()
    train_dataloader, val_dataloader = produce_dataloaders(config,
                                                           per_device_train_batch_size,
                                                           per_device_eval_batch_size)
    model = produce_model(config)
    optimizer = produce_optimizer(model, learning_rate)

    num_update_steps_per_epoch, max_train_steps, num_training_epochs = \
        calculate_train_details(train_dataloader,
                                gradient_accumulation_steps,
                                num_train_epochs)

    lr_scheduler = produce_lr_scheduler(optimizer,
                                        num_warmup_steps,
                                        gradient_accumulation_steps,
                                        max_train_steps)
    
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch, max_train_steps, num_training_epochs = \
        calculate_train_details(train_dataloader,
                                gradient_accumulation_steps,
                                num_train_epochs)

    total_batch_size = \
            per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    checkpoint_path = setup_checkpointing(checkpoint_dir)
    if start_epoch != 0:
        resume_from_checkpoint(checkpoint_path, start_epoch, accelerator)

    progress_bar = tqdm(range(max_train_steps),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                        disable=not accelerator.is_local_main_process)
                    
    run_training(start_epoch,
               num_training_epochs,
               model,
               train_dataloader,
               val_dataloader,
               accelerator,
               optimizer,
               lr_scheduler,
               per_device_eval_batch_size,
               max_train_steps,
               checkpoint_path,
               progress_bar)

if __name__ == '__main__':
    main()
