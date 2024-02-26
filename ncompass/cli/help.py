import sys
import ncompass.client.functional as F
from ncompass.network_utils import exec_url

def help_msg():
    print('\t nccli-help :: Prints this help message.')

def start_session_msg():
    print('\t nccli-start-session <api_key> :: Starts the session for an api_key')

def stop_session_msg():
    print('\t nccli-stop-session <api_key> :: Stops the session for an api_key')

def help():
    print('Usage instructions for the nCompass cli (nccli-):')
    help_msg()
    start_session_msg()
    stop_session_msg()

def main():
    help()
