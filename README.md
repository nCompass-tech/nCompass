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
We strive to make sure that all our capabilities can be exposed to you with as close to one line
of code as possible. Below is an example of how you would do that with us for streaming prompt 
complettion. The following lines of code and other examples like it can be found in the
examples/hello_world.py file.
```
from ncompass.client import nCompassOLOC

prompt = 'Give me 5 tools I can use to accelerate inference of my ML model?'
api_key = '<api_key>'
nCompassOLOC().start_session(api_key)
for i in range(3):
    nCompassOLOC().complete_prompt(api_key, prompt, max_tokens = 300, temperature = 0.5
                                   , top_p = 0.9, stream = True, pprint = True)
nCompassOLOC().stop_session(api_key)
```
The section below unwraps the code a bit more. It is still simple, but not quite OLOC.

## More fine grained control 
The following code (also found in *examples/fine_grained_control.py*) is an example of using the 
nCompass client with more fine grained control. For instance: 
- you could replace `client.wait_until_model_running()` with the non-blocking version 
  `client.model_is_running()` which just returns a boolean. 
- use a custom way to iterate through the response by replacing `client.print_prompt`. The
  *examples/hello_world.py* file has an example (*use_streaming_iterator*) of extracting the 
  iterator from our OLOC version as well
```
from ncompass.client import nCompass

# The client will automatically read the API key from the environment variable passed here.
# Alternatively, you can directly pass the api_key to the client (see README.md)
client = nCompass(custom_env_var = 'NCOMPASS_API_KEY')
client.start_session()
client.wait_until_model_running()
params = {'max_tokens':    300 # max output tokens requested
          , 'temperature': 0.5
          , 'top_p':       0.9
          , 'stream':      True}
prompt = 'Give me 5 tools I can use to accelerate inference of my ML model?'
response_iterator = client.complete_prompt(prompt, **params)
client.print_prompt(response_iterator)
client.stop_session()
```
Note that here we read the API key from a custom environment variable NCOMPASS_API_KEY which you
can set with the token we provide you. However, the API also works if you directly provide the API
key to the client directly as below.
```
client = nCompass(api_key = '<key we provide you>')
```

## **IMPORTANT:** What are sessions?
Sessions are how nCompass keeps track of usage statistics. nCompass billing is **not per_token** 
but rather **time_based**. This means, your costs do not scale linearly with your inputs. 
We simply charge you for the time between calls to start and stop session. 

There are two ways to start and stop sessions:
- **Using the command line**: When you install the library via pip, you will have 3 command line
  calls available to you. 
  - `nccli-help` : prints a help msg on how to use the cli commands
  - `nccli-start-session <api_key>` : starts a session for that api_key
  - `nccli-stop-session <api_key>` : stops a session for that api_key

- **Programatically**: In all the examples above, the `start_session` and `stop_session` calls
  perform the tasks of starting and stopping sessions.

**Please do not forget to stop sessions you have started as billing occurs between the start and
stop of a session.**

## Note on thread safety
nCompassOLOC client is **not thread safe**, but the nCompass client **is thread safe**. 
So if you would like to use the client inside threads that you are spawning (for eg. when using
a wsgi web frameworks like *flask*), then please only use the nCompass client.

## Chat Completions
The above examples focused on simple prompt completions. If you would like to use chat based
completions with a custom template, please checkout the following example.

```
client = nCompass(api_key = <api_key>)
client.start_session()
client.wait_until_model_running()
params = {'max_tokens':    300 # max output tokens requested
          , 'temperature': 0.5
          , 'top_p':       0.9
          , 'stream':      True}
messages = [
    {
        "role": "system",
        "content": "You are the nCompass assistant, you are an expert in accelerating ML inference.",
    },
    {
        "role": "assistant",
        "content": "Hello, how can I speed up your ML inference today?",
    },
    {
        "role": "user",
        "content": "What tweaks can I make to my model to make it faster?",
    },
    {
        "role": "assistant",
        "content": "Ah, the age old question.",
    },
]
response_iterator = client.complete_chat(messages=messages
                                         , add_generation_prompt=True
                                         , **params)
client.print_prompt(response_iterator)
client.stop_session()
```
The main changes from prompt completion is that we now use the `complete_chat` API instead of the
`complete_prompt` API. For convenience, there is the equivalent `nCompassOLOC().complete_chat(...)`
API as well. Also the argument name is no longer *prompt*, but rather *messages*.
*add_generation_prompt* is also a boolean that can be passed through.

The chat_template used is the chat_template specified in the model's tokenizer (if HF tokenizer is
used). If you have a custom template you would like to pass through, there is support for that 
that is in testing, so please contact either `aditya.rajagopal@ncompass.tech` or
`diederik.vink@ncompass.tech` for information.

## Batch Requests
For both completions and chat, we support batch requests which is enabled by setting the 
*stream* argument in params to False. 
For *prompt*, the input can either be a single string or a list of strings.
For *messages*, the input can either be a single list or a list of lists.
