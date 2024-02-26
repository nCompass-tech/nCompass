from ncompass.client import nCompass

def fine_grained_control():
    # The client will automatically read the API key from the environment variable passed here.
    client = nCompass(custom_env_var = 'NCOMPASS_API_KEY')
    # Starts an execution session
    client.start_session()
    # Blocks until the model is ready to receive a prompt and run it
    client.wait_until_model_running()
    params = {'max_tokens':    300 # max output tokens requested
              , 'temperature': 0.5
              , 'top_p':       0.9
              , 'stream':      True}
    prompt = 'Give me 5 tools I can use to accelerate inference of my ML model?'
    # complete_prompt returns a streaming iterator that can then be read in a loop to extract the
    # response
    response_iterator = client.complete_prompt(prompt, **params)
    client.print_prompt(response_iterator)
    # Stops an execution session
    client.stop_session()

fine_grained_control()
