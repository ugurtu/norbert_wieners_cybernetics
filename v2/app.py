from flask import Flask, render_template, request, jsonify
import chatbot

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):

    # Let's chat for 5 lines
    return chatbot.retrieve_information(text)

"""
if __name__ == '__main__':
    app.run()
"""