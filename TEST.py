import time

from utils.server_utils import send_move_request


while True:
    send_move_request(input("Enter column (0-6): ").strip())
    time.sleep(1)