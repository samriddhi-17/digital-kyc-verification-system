from flask import Flask, render_template, request
import requests
import base64

app = Flask(__name__)

BACKEND_URL = "http://127.0.0.1:8000/verify_kyc"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit_kyc", methods=["POST"])
def submit_kyc():
    user_id = request.form["user_id"]
    address_doc_type = request.form["address_doc_type"]
    identity_doc_type = request.form["identity_doc_type"]

    # Selfie (Base64)
    selfie_b64 = request.form["selfie_captured"]
    header, encoded = selfie_b64.split(",", 1)
    selfie_bytes = base64.b64decode(encoded)

    selfie_file = ("selfie.jpg", selfie_bytes, "image/jpeg")

    files = {
        "address_proof": (
            request.files["address_proof"].filename,
            request.files["address_proof"].read(),
            request.files["address_proof"].content_type,
        ),
        "identity_proof": (
            request.files["identity_proof"].filename,
            request.files["identity_proof"].read(),
            request.files["identity_proof"].content_type,
        ),
        "selfie": selfie_file
    }

    data = {
        "user_id": user_id,
        "address_doc_type": address_doc_type,
        "identity_doc_type": identity_doc_type
    }

    response = requests.post(BACKEND_URL, files=files, data=data)
    result = response.json()

    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
