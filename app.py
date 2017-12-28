from flask import Flask,render_template,request,redirect,url_for
import model
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results", methods=['GET','POST'])
def results():
    if request.method == 'POST':
        try:
            test1 = float(request.form['data1'])
            test2 = float(request.form['data2'])
            input_data = np.array([test1,test2])
        except:
            return "Invalid input"
        results = model.predict(app.graph,input_data)
        return str(results)
    return redirect(url_for('index'))

app.graph = model.load_graph('static/model.sav') 

if __name__ == "__main__":
    app.run(debug=True)    