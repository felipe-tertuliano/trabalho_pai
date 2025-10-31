from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Flask App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input, select, button {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #e9f7ef;
            border-radius: 5px;
            display: none;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Interactive Flask Interface</h1>
        
        <!-- Form 1: Text Processing -->
        <div class="form-group">
            <h3>Text Processor</h3>
            <label for="textInput">Enter text:</label>
            <input type="text" id="textInput" placeholder="Type something...">
            <button onclick="processText()">Process Text</button>
        </div>

        <!-- Form 2: Calculator -->
        <div class="form-group">
            <h3>Calculator</h3>
            <label for="num1">Number 1:</label>
            <input type="number" id="num1" value="0">
            <label for="num2">Number 2:</label>
            <input type="number" id="num2" value="0">
            <label for="operation">Operation:</label>
            <select id="operation">
                <option value="add">Add</option>
                <option value="subtract">Subtract</option>
                <option value="multiply">Multiply</option>
                <option value="divide">Divide</option>
            </select>
            <button onclick="calculate()">Calculate</button>
        </div>

        <!-- Form 3: Data Display -->
        <div class="form-group">
            <h3>Get Server Data</h3>
            <button onclick="getServerData()">Get Current Server Info</button>
        </div>

        <!-- Results Area -->
        <div id="result" class="result"></div>
    </div>

    <script>
        function processText() {
            const text = document.getElementById('textInput').value;
            fetch('/process-text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({text: text})
            })
            .then(response => response.json())
            .then(data => {
                showResult('Text processed: ' + data.processed_text);
            })
            .catch(error => {
                showResult('Error: ' + error, true);
            });
        }

        function calculate() {
            const num1 = parseFloat(document.getElementById('num1').value);
            const num2 = parseFloat(document.getElementById('num2').value);
            const operation = document.getElementById('operation').value;
            
            fetch('/calculate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    num1: num1,
                    num2: num2,
                    operation: operation
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showResult('Error: ' + data.error, true);
                } else {
                    showResult(`Result: ${num1} ${data.operation_symbol} ${num2} = ${data.result}`);
                }
            })
            .catch(error => {
                showResult('Error: ' + error, true);
            });
        }

        function getServerData() {
            fetch('/server-info')
            .then(response => response.json())
            .then(data => {
                showResult(`Server Time: ${data.time}<br>Random Number: ${data.random_number}`);
            })
            .catch(error => {
                showResult('Error: ' + error, true);
            });
        }

        function showResult(message, isError = false) {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = message;
            resultDiv.style.display = 'block';
            resultDiv.className = isError ? 'result error' : 'result';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/process-text', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Process the text (example: reverse and uppercase)
    processed_text = text[::-1].upper()  # Reverse and convert to uppercase
    
    return jsonify({
        'original_text': text,
        'processed_text': processed_text
    })

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    num1 = data.get('num1', 0)
    num2 = data.get('num2', 0)
    operation = data.get('operation', 'add')
    
    try:
        if operation == 'add':
            result = num1 + num2
            symbol = '+'
        elif operation == 'subtract':
            result = num1 - num2
            symbol = '-'
        elif operation == 'multiply':
            result = num1 * num2
            symbol = '×'
        elif operation == 'divide':
            if num2 == 0:
                return jsonify({'error': 'Division by zero'}), 400
            result = num1 / num2
            symbol = '÷'
        else:
            return jsonify({'error': 'Invalid operation'}), 400
            
        return jsonify({
            'result': result,
            'operation_symbol': symbol
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/server-info')
def server_info():
    import datetime
    import random
    
    return jsonify({
        'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'random_number': random.randint(1, 100)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)