import requests

SERVER_URL = "http://192.168.206.132:5000/move"  # Replace with your local IP


def send_move_request(column):
    response = requests.post(SERVER_URL, json={"column": column})
    print(response.json())

if __name__ == "__main__":
    while True:
        col = input("Enter column (0-6) or 'q' to quit: ").strip()
        if col.lower() == 'q':
            break
        try:
            col = int(col)
            send_move_request(col)
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 6.")
