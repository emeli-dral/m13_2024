from flask import Flask, render_template, request
from topic_model import TopicClassifier

app = Flask(__name__)

classifier = TopicClassifier()

@app.route("/classify", methods=["POST", "GET"])
def index_page(text="", result=""):
    if request.method == "POST":
        text = request.form["text"]
        result = classifier.predict(text)
	
    return render_template('index.html', text=text, result=result)


if __name__ == "__main__":
    app.run(port=5005)