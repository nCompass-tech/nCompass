from ncompass.client import nCompass

# The client will automatically read the API key from the environment variable passed here.
client = nCompass(custom_env_var = 'MISTRAL_7B_API_KEY')
client.wait_until_model_running()
params = {'max_tokens':    300 # max output tokens requested
          , 'temperature': 0.5
          , 'top_p':       0.9
          , 'stream':      True}
prompt = 'Give me 5 tools I can use to accelerate inference of my ML model?'
response_iterator = client.complete_prompt(prompt, **params)
ttft = client.print_prompt(response_iterator)
print(f'ttft = {ttft*1000:.2f}ms')
