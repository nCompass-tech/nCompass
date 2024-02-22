# ncompass
A One-Line-Of-Code (OLOC) API that makes it easy for anyone to access low-latency open-source or
custom AI models. This repo has some examples of the various ways in which you can use our API. 

## Installing
You can install our package via pip with the following command:  
```
pip install ncompass
```

## Getting your API Key
We're working on automating the process for you to select a model of your choice and getting an API
key, but for now please email either aditya.rajagopal@ncompass.tech or diederik.vink@ncompass.tech to
be onboarded and provided with your API key. Thanks!

## One-Line-Of-Code (OLOC)
We strive to make sure that all our capabilities can truly be exposed to you with exactly one line
of code. Below is an example of how you would do that with us for streaming prompt complettion.
```
from ncompass.client import nCompassOLOC

prompt = 'Give me 5 tools I can use to accelerate inference of my ML model?'
for i in range(3):
    ttft = nCompassOLOC().complete_prompt('<api_key>', prompt, max_tokens = 300, temperature = 0.5
                                          , top_p = 0.9, stream = True, pprint = True)
    print(f'ttft = {ttft*1000:.2f}ms')
```
The section below unwraps the code a bit more. It is still simple, but not quite OLOC'd.

## More fine grained control 
The following code (also found in examples/hello_world.py) sets up an nCompass client, parameters 
and calls the complete_prompt API. Our API returns a streaming iterator (response_iterator) and 
we also provide a print_prompt function which can print out the stream.
```
from ncompass.client import nCompass

# The client will automatically read the API key from the environment variable passed here.
client = nCompass(custom_env_var = 'NCOMPASS_API_KEY')
client.wait_until_model_running()
params = {'max_tokens':    300 # max output tokens requested
          , 'temperature': 0.5
          , 'top_p':       0.9
          , 'stream':      True}
prompt = 'Give me 5 tools I can use to accelerate inference of my ML model?'
response_iterator = client.complete_prompt(prompt, params)
ttft = client.print_prompt(response_iterator)
print(f'ttft = {ttft*1000:.2f}ms')
```

Note that here we read the API key from a custom environment variable NCOMPASS_API_KEY which you
can set with the token we provide you. However, the API also works if you directly provide the API
key to the client directly as below.
```
client = nCompass(api_key = '<key we provide you>')
```
