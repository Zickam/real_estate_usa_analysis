# Setup Envirnomnet
echo "Creating python env"
cd /var/lib/jenkins/workspace/download
python3 -m venv .venv #создать виртуальное окружение в папку
. .venv/bin/activate   #активировать виртуальное окружение
pip install setuptools
pip install ipykernel --upgrade
pip install pandas
pip install scikit-learn fastapi uvicorn
pip list
echo "Python env created"


# Download
echo "Downloading data"
wget -O "realtor-data.zip.csv.zip" "https://storage.googleapis.com/kaggle-data-sets/3202774/7981839/compressed/realtor-data.zip.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250403%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250403T042359Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=0f990a735c2c5dba8400dbeb9a234ce16ce355e560ea8402c9e52bf4d513ce6eca1026e55c6aae14be023677a28a4182ef78a5416d9c783e894b941c1a52507c2941666fe36ebe2d343ff14237c408906560dc306adc11f7180c1ed37bdedf9a4338af0479d55422fb4b566938eee7e7a22ed34fe367aaf31ebe6cedd04a15a022ae83586cb7d4f81dfd75792c6946f4fb3e7fded3894e6529592ba364863d682614216853107c1f179ba3b11929b981622ac8cac35a422e2134ca58edb09b42e81518c668d2d2d7dde62935d252fd32ed16498a9b9c95d16fe3ce362968089371835a1423e0d52de5b1e0fbe95206b48e23929bc1cc019ed78439fabcab264c"
unzip -o "realtor-data.zip.csv.zip"
mv "realtor-data.zip.csv" "realtor-data.csv"
echo "Data downloaded"


# Train
echo "Start clean and train"
cd /var/lib/jenkins/workspace/download
. .venv/bin/activate
python3 clean_and_train.py
echo "End clean and train"


# Deploy
echo "Start webserver"
cd /var/lib/jenkins/workspace/download
export BUILD_ID=dontKillMe
export JENKINS_NODE_COOKIE=dontKillMe
. .venv/bin/activate
python3 webserver.py
sleep 10
echo "End webserver"


# Healtcheck
curl -X 'GET'   'http://127.0.0.1:8002/predict'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
"brokered_by": "103378.0",      "status": "for_sale",   "price": 205000.0,      "bed": 3.0,     "bath": 2.0,    "acre_lot": 0.12,       "street": 1962661.0,    "city": "Adjuntas",     "state": "Puerto Rico", "zip_code": 601.0,      "house_size": 920.0,    "prev_sold_date": "null"}'


# Pipeline
pipeline {
    agent any

    stages {
        stage('Setup Environment') {
            steps {

                build job: "setup_env"

            }
        }

        stage('Start Download') {
            steps {

                build job: "download"

            }
        }

        stage ('Train') {

            steps {

                script {
                    dir('/var/lib/jenkins/workspace/download') {
                        build job: "train"
                    }
                }

            }
        }

        stage ('Deploy') {
            steps {
                build job: 'deploy'
            }
        }

        stage ('Status') {
            steps {
                build job: 'healthcheck'
            }
        }
    }
}

