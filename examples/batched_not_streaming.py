from ncompass.client import nCompassOLOC

# Example of batched response instead of streaming.

def batched_not_streaming(API_KEY):
    try:
        nCompassOLOC().start_session(API_KEY)
        prompt = 'What does the fox say?'
        response = nCompassOLOC().complete_prompt(API_KEY
                                                  , prompt
                                                  , max_tokens = 300
                                                  , temperature = 0.5
                                                  , top_p = 0.9
                                                  , stream = False # this sets batch processing
                                                  )
        print(response)
        nCompassOLOC().stop_session(API_KEY)
    except Exception as e:
        print(str(e))

batched_not_streaming('<api_key>')

