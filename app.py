 #main.py

from flask import Flask
from model import video as Vd
from flask_cors import CORS, cross_origin

app = Flask(__name__)

cors = CORS(app, resources={r"/": {"origins": "http://localhost:port"}})

@app.route("/")
def beging():
    return """
    <html> 
        <body>
                <h1> AI - Visão computacional: </h1>
                    <u1>
                        <a href= "/data"> Iniciar Webcam </a>

                    

        </body> 
    
    </html> """

@app.route("/data", methods=['GET'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def data():
     
      v = Vd()
        
        
      return jsonify(
        gesto = v
        
    )
   
if __name__ == '__main__':

    app.run(host="127.0.0.1", port=5000, debug =True)


