from ncompass.client import nCompassOLOC

# Example if you would like to run and print the prompt at once (setting pprint=True)
def run_oloc():
    prompt = 'Give me 5 tools I can use to accelerate inference of my ML model?'
    for i in range(3):
        ttft = nCompassOLOC().complete_prompt(get_api_key(), prompt, max_tokens = 300, temperature = 0.5
                                              , top_p = 0.9, stream = True, pprint = True)
        print(f'ttft = {ttft*1000:.2f}ms')

# Example if you would like to get an iterator and then print the prompt (setting pprint=False)
def run_oloc_iterator():
    prompt = 'Give me 5 tools I can use to accelerate inference of my ML model?'
    for i in range(3):
        # setting pprint to False returns a streaming iterator
        iterator = nCompassOLOC().complete_prompt(get_api_key(), prompt, max_tokens = 300, temperature = 0.5
                                                  , top_p = 0.9, stream = True, pprint = False)
        
        # if you would like to print the iterator
        ttft = nCompassOLOC().print_response(iterator)
        print(f'ttft = {ttft*1000:.2f}ms')
