from web_api import app as main_app
import uvicorn

app = main_app

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
