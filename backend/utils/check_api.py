import os
import requests


def check_api_health(api_url):
    health_url = f"{api_url.rstrip('/')}/health"
    try:
        response = requests.get(health_url)
        if response.status_code == 200:
            print(f"API at {api_url} is healthy.")
            return True
        else:
            print(f"API at {api_url} returned status code {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Failed to connect to API at {api_url}: {str(e)}")
        return False


# Get the API URL from the environment variable or use a default
PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")

if __name__ == "__main__":
    if check_api_health(PREFECT_API_URL):
        print("API is healthy. Proceeding with flow registration.")
    else:
        print("API is not healthy. Please check your Prefect server and API URL.")
        exit(1)
