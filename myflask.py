from model import SimpleLSTM
import torch
from flask import Flask, request, jsonify
import json


input_size = 1086
hidden_size = 124
output_size = 9
model = SimpleLSTM(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('checkpoint_last_seed-1.pth')['model'])
label_map = {'Chào': 0,'chúc tết': 1,'Cứu': 2,'dừng lại': 3,'nguy hiểm': 4,'tạm biệt': 5,'tết âm': 6,'đoàn tụ': 7,'đọc sách': 8}
def predict(input):
    model.eval()
    input = input.unsqueeze(0)
    # Perform inference on the sample
    with torch.no_grad():
        output = model(input)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    predicted_class_string = [key for key, value in label_map.items() if value == predicted_class][0]
    return predicted_class_string

input = torch.rand((124,164))
predict(input)
app = Flask(__name__)


@app.route('/')
def welcom():
    return "hello"
    
@app.route('/predict', methods=['POST'])
def get_prediction():
    try:
        data = request.json
        input_data = torch.tensor(data['input'])  # Assuming you'll send input data in the request body as JSON
        result = predict(input_data)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')