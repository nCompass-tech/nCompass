from ncompass.client import nCompassOLOC

# Examples on how to programatically start a session, run a prompt and stop a session 
# You are only billed for the time you spend between a start and stop session.

def hello_world(API_KEY):
    prompt = 'How do I make pizza?'
    nCompassOLOC().start_session(API_KEY)
    nCompassOLOC().complete_prompt(API_KEY, prompt, max_tokens = 300, temperature = 0.5
                                   , top_p = 0.9, stream = True, pprint = True)
    nCompassOLOC().stop_session(API_KEY)

def run_multiple_prompts(API_KEY):
    try:
        # NOTE: if you try to run complete_prompt without starting a session, the program will
        # throw an exception
        nCompassOLOC().start_session(API_KEY)
        
        # Example program that goes over multiple user interactions... 
        for i in range(3):
            # custom logic to setup prompt...
            prompt = 'Give me 5 tools I can use to accelerate inference of my ML model?'
            # print prompt to user
            nCompassOLOC().complete_prompt(API_KEY, prompt, max_tokens = 300, temperature = 0.5
                                           , top_p = 0.9, stream = True, pprint = True)
            # custom logic to handle user input ...
        
        # NOTE: if you forget to stop a started session, your billing continues till the
        # session is stopped. Please do not forget to do so!  
        nCompassOLOC().stop_session(API_KEY)
    except Exception as e:
        print(str(e))

# Example if you would like to get an iterator and then print the response (setting pprint=False)
def use_streaming_iterator(API_KEY):
    nCompassOLOC().start_session(API_KEY)
    prompt = 'Give me 5 tools I can use to accelerate inference of my ML model?'
    for i in range(3):
        # setting pprint to False returns a streaming iterator
        iterator = nCompassOLOC().complete_prompt(API_KEY, prompt, max_tokens = 300, temperature = 0.5
                                                  , top_p = 0.9, stream = True, pprint = False)
        
        # if you would like to print the iterator
        nCompassOLOC().print_response(iterator)
    nCompassOLOC().stop_session(API_KEY)

api_key = '<api_key>'
hello_world(api_key)
run_multiple_prompts(api_key)
use_streaming_iterator(api_key)
