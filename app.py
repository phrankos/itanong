from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('layout.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        data = request.get_json()
        user_query = data['query']
        
        # Call the function from the Advanced_Query_Pipelines_over_Tabular_Data.py file
        # to get the response
        response_message = get_answer(user_query)
        # Placeholder response
        response_message = f"Response: {user_query}"
        
        return jsonify({"response": response_message})
    
    return render_template('chatbot.html')

if __name__ == '__main__':
    app.run(debug=True)
