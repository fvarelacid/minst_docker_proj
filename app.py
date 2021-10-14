from flask import Flask, render_template, request
from handgestures.transform_image import transform_single_image
from handgestures.main import Net
import torch
import numpy as np


app = Flask(__name__)

@app.route('/', methods=['GET'])

def hello_world():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    
    #Transform the Image into a Skeleton
    img = transform_single_image(image_path)
    img = torch.Tensor(img)

    #import the model from RPS_net.pth
    save_path = 'handgestures/RPS_net.pth'
    # model = Net()
    model = Net()
    model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    model.eval()

    # Generate prediction
    rps_class = model(img.reshape(1, 1, 128, 128))

    # Predicted class value using argmax
    predicted_class = np.argmax(rps_class.detach().numpy())

    if predicted_class == 0:
        hand_result = 'rock'
    if predicted_class == 1:
        hand_result = 'paper'
    if predicted_class == 2:
        hand_result = 'scissors'

    classification = 'This gesture is: ' + hand_result

    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

