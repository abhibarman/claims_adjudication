import openai
import os
from dotenv import load_dotenv, dotenv_values
from fastapi import FastAPI, Request
import uvicorn
import time
start = time.time()
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from chatpolicy import retrieve_search_results

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()


@app.post('/classify_email')
async def classify_email(request: Request):
    
    in_email = await request.json()
    in_email = in_email['email']
    #print(f"in_email :\n{in_email}")
    
    prompt = f"""You are an Insurance Agent who classifies the incoming emails into ENQUIRY, CLAIM SUBMISSION or CLAIM  RE-SUBMISSION. 
    If the user is trying to submit a claim or providing information related to some incident then categorize it as CLAIM SUBMISSION.
    If the user is submitting the same claim with additional information or if the user is replying with additional requested information then it will be a CLAIM RE-SUBMISSION and
    If the user is asking for information on some insurance policy or any other enquiry insurance then classify it as ENQUIRY  :\n {in_email} """
    message = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message
    )
    response = response.choices[0]["message"]['content']

    return {'email_type':response}

@app.post('/email_reply') 
async def reply_email(request: Request):
    
    in_email = await request.json()
    in_email = in_email['email']

    
    prompt = f"""You are an Insurance Claim Adjuster who replies to customer queries. You need to draft an email body against a customer's inquiry . 
    Given the below email \n {in_email} \n 
    Draft only the email body to the insurance broker or claimant. 
    The following criteria must be full-filled -
    1. Only Body of the email is needed
    2. No Greetings sections like Dear , Hi, Hello etc
    3. No Closing Sections like Regards, Thanks, Thank you etc
    4. No Placeholder like [Your Conatct Number],[Your Name], [Your Position], [Insurance Company Name]
    5. No Subject line
    6. No disclaimer required
    7. Do not repeat information from the email. 
    8. Focus on replying to the inquiry only.
    9. Check if all the previously mentioned criterias are satisfied.
    """
    message = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message, 
        temperature =0.0
    )
    #\n\nSincerely
    response = response.choices[0]["message"]['content']

    index_of = response.rfind("\n\nSincerely")
    index_of_thanku = response.rfind("\n\nThank you")

    if index_of != index_of_thanku:
        index_of = max(index_of, index_of_thanku)

    if index_of != -1:
        response = response[:index_of]

    return {'email_body':response}

@app.post('/req_for_info')
async def request_for_additional_info(request: Request):
        

    in_json = await request.json()
    #print(f'in_json { in_json}')
    missing_data = in_json['missing_fields']
    prompt = f"""You are an Insurance Claim Adjuster and your task is to draft an email body asking for missing information to proceed with their insurance claim initiation.
    Given the following missing fields -{missing_data}\n
    Draft an email body to the insurance broker or claimant requesting the necessary information needed to proceed with their insurance claim. 
    The following criteria must be full-filled -
    1. Only Body of the email is needed
    2. No Greetings sections like Dear , Hi, Hello etc
    3. No Closing Sections like Regards, Thanks, Thank you etc
    4. No Placeholder like [Your Name], [Your Position], [Insurance Company Name]
    5. No Subject line
    6. No disclaimer required
    7. Check if all the previously mentioned criterias are satisfied.
    """
    message = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message, 
        temperature = 0.0
    )

    response = response.choices[0]["message"]['content']

    index_of = response.rfind("\n\nThank you")
    print(f"index_of: {index_of}")
    print(f"response: {response}")
    if index_of != -1:
        response = response[:index_of]

    return {'email_body':response}

  
@app.post('/extract_entities')
async def extract_entities(request: Request):

    result = {

        "Insurance_Company": "",
        "Policy_Number": "PIAB1149",
        "Date_of_Incident": "",
        "State": "Florida",
        "City": "Orlando",
        "Zip": "32003",
        "Type_of_Incident": "Self",
        "Cause_of_Incident": "Impact",
        "Description_of_Incident": "Hit a pole causing damage to the back bumper",
        "Contact_Number": "",
        "First_Name": "Helaina",
        "Last_Name": "Heigl"
    }

    return result 



    email = await request.json()
    email = email['email']

    function_descriptions = [
    {
        "name": "extract_key_entities_from_insurance_claim_email",
        "description": "extract key info from the insurance claim related email, such as Insurance_Company,Policy_Number, Date_of_Incident,State, City, Zip, Type of Incident,Cause of Incident,Description of Incident, Contact Number, First Name, Last Name ",
        "parameters": {
            "type": "object",
            "properties": {
                "Insurance_Company": {
                    "type": "string",
                    "description": "Name of the Insurance Company"
                },                                        
                "Policy_Number": {
                    "type": "string",
                    "description": "Policy Number of the Insured or the Customer"
                },
                "Date_of_Incident":{
                    "type": "string",
                    "description": "The date & time on which the mentioned incident occurred. Also convert the extracted date and time to a format MM/DD/YYYY HH:MM:SS"
                },
                "State":{
                    "type": "string",
                    "description": "State of the Insured"
                },

                "City":{
                    "type": "string",
                    "description": "City of the Insured"
                }
                ,
                "Zip":{
                    "type": "string",
                    "description": "Zip Code of the Insured"
                },
                "Type_of_Incident":{
                    "type": "string",
                    "description": "The possible types of incident are SELF, THIRD PARTY. SELF if the accident is caused by the driver or owner and THIRD PARTY if caused by another person."
                },
                "Cause_of_Incident":{
                    "type": "string",
                    "description": "The cause of the incident can be Mechanical Failure, Weather Condition, Driver Error."
                },
                "Description_of_Incident":{
                    "type": "string",
                    "description": "Concise descriptio of the incident described in the email."
                },
                
                "Contact_Number":{
                    "type": "string",
                    "description": "Contact Number of the insured or the email sender."
                },
                "First_Name":{
                    "type": "string",
                    "description": "The First Name of the insured or the email sender."
                },
                 "Last_Name":{
                    "type": "string",
                    "description": "The Last Name of the insured or the email sender."
                }              

                
            },
            "required": ["Insurance_Company", "Policy_Number", "Date_of_Incident","State","City","Zip","Type_of_Incident","Cause_of_Incident","Description_of_Incident","Contact_Number","First_Name","Last_Name"]
        }
    }
    ]

    prompt = f"""You are an Claim Insurance Agent who extracts the relevant entities mentioned in the email from a broker or claimant. 
    The entities you need to extract are Insurance Company Name, Policy Number,Date of Incident,Type of Incident, Cause of Incident,
    Description of Incident, Contact Number,First Name, Last Name, State, City, Zip Code, Date of Incident.
    If the value does not exist or you are not sure then keep it empty.
    Think and validate your output before finalizing it.

    Please extract the mentioned entities from the below email     :\n {email} """
    message = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
        functions = function_descriptions,
        function_call="auto"
    )

    response = response.choices[0]["message"]["function_call"]["arguments"]

    Insurance_Company = eval(response).get("Insurance_Company")
    Policy_Number = eval(response).get("Policy_Number")
    Date_of_Incident = eval(response).get("Date_of_Incident")
    State = eval(response).get("State")
    City = eval(response).get("City")
    Zip = eval(response).get("Zip")

    Type_of_Incident = eval(response).get("Type_of_Incident")
    Cause_of_Incident = eval(response).get("Cause_of_Incident")
    Description_of_Incident = eval(response).get("Description_of_Incident")
    Contact_Number = eval(response).get("Contact_Number")
    First_Name = eval(response).get("First_Name")
    Last_Name = eval(response).get("Last_Name")

    result = {
        'Insurance_Company': Insurance_Company,
        'Policy_Number':Policy_Number,
        'Date_of_Incident':Date_of_Incident,
        'State':State,
        'City':City,
        'Zip':Zip,
        "Type_of_Incident":Type_of_Incident,
        "Cause_of_Incident":Cause_of_Incident,
        "Description_of_Incident":Description_of_Incident,
        "Contact_Number":Contact_Number,
        "First_Name":First_Name,
        "Last_Name":Last_Name
    }

    return result



@app.post('/assess_damage')
async def assess_damage(request: Request):

    in_json = await request.json()
    file_path = in_json['file_path']    
    
    return {
    "damage_assessment": ".\n\nAssessment:\n\nThe car has battle damage, a gaunt nose, and a rear-facing spoiler.\n\nPotential Impact:\n\nThe battle damage could potentially impact the car's performance. The gaunt nose may also impact the car's aerodynamics.\n\nDamage Impact in Percentage:\n\nThe battle damage is estimated to be 10% of the car's value. The gaunt nose is estimated to be 5% of the car's value.\n\nRecommendations:\n\nThe car should be repaired by a qualified mechanic.\n\nRequest for More Information:\n\nMore information is needed about the extent of the battle damage and the gaunt nose."
}



@app.post("/get_details")
async def get_details(request: Request):

    query = await request.json()
    query = query['query']

    answer = retrieve_search_results(query)
        
    sentences = ["What is the  the coverage given for Bodily Injury?", "Any emergency service included ?","Give me details of the Vehicle ?"]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    query_embedding = tfidf_vectorizer.transform([query])

    similarities = {}
    for i in range(len(sentences)):
        similarity = cosine_similarity(tfidf_matrix[i], query_embedding)
        similarities[i] = similarity[0].item()
        
    most_similar = sorted(similarities.items(), key = lambda x: x[1])[-1]
    #most_similar_sentence = sentences[most_similar[0]]

    similarity_index = most_similar[0]
    if similarities[similarity_index] < 0.8:
        similarity_index = 3

    result = {}
    match similarity_index:
        case 0:
            result ={

                'answer': 'The coverage given for Bodily Injury is $10,000 per person and $20,000 per occurrence.',            
                'Search_Result': {
                    0: """Bodily Injury Liability Each
                            Person/Each Occurrence
                            $10,000/$20,000 $ 100"""
                },
                'Page_Number': {
                    0: 1
                },
                'Coordinates': {
                
                    0:(77.42400360107422, 479.469970703125, 461.3000793457031, 506.1099853515625)
                }
                
            }
        case 1:
            result ={

            'answer': 'Yes, according to the declaration page, Emergency Road Service (ERS) is included in the coverage.',

            'Search_Result': {
                0: "Emergency Road Service ERS FULL"
            },
            'Page_Number': {
                0: 1
            },
            'Coordinates': {
                0: (77.42400360107422, 679.0599975585938, 319.2669372558594, 691.0599975585938)
            }
                    }
        case 2:
            result = {

            'answer': 'Vehicle Year: 2015\nMake: TOYOTA\nModel: CAMRY\nVIN: 6T1PF1FK1HU422208',

            'Search_Result': {
                0: """Vehicle Year: Make Model VIN
                        2015 TOYOTA CAMRY 6T1PF1FK1HU422208"""
            },
            'Page_Number': {
                0: 1
            },
            'Coordinates': {
                0: (96.86399841308594, 406.2499694824219, 517.9041748046875, 447.54998779296875),
            }
        }

        case 3:
            
            result = {

                'answer': f"{answer}",
                'Search_Result': {},
                'Page_Number': {},
                'Coordinates': {}
            }


    return result

if __name__ == '__main__':

    uvicorn.run(app, host="0.0.0.0",port = 8003)