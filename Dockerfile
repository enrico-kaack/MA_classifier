FROM python:3.8-slim
WORKDIR /classification
RUN apt-get update && apt-get install gcc python3-dev unzip -y 

#install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copy the rest
COPY . ./

#extract all datasets
RUN unzip datasets/final_dataset.zip && unzip datasets/final_test.zip && unzip datasets/final_validation.zip

#make entrypoint scripts executeable
RUN chmod +x run_everything.sh
CMD [ "./run_everything.sh" ]


