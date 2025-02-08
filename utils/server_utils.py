import threading

import requests

SERVER_URL = "http://192.168.206.132:5000/move"  # Replace with your local IP

def send_move_request(column):
    try:
        print(f"Sending move request for column {column}")
        # Send the request; we don't care about the response
        requests.post(SERVER_URL, json={"column": column})
        print(f"Request sent for column {column}")
    except requests.exceptions.RequestException:
        print("Failed to send request. Is the server running?")

def send_move_request_async(column):
    # Create and start a daemon thread so it doesn't block program exit
    thread = threading.Thread(target=send_move_request, args=(column,), daemon=True)
    thread.start()
    
if __name__ == "__main__":
    while True:
        col = input("Enter column (0-6) or 'q' to quit: ").strip()
        if col.lower() == 'q':
            break
        try:
            col = int(col)
            send_move_request_async(col)
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 6.")
