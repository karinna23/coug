from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from .models import *
from .forms import PredictionForm
import joblib
import numpy as np

# ----- WEASYPRINT PDF -----
from django.http import HttpResponse
from django.template import loader
from weasyprint import HTML
import datetime

# from .forms import RegistrationForm


# def login_view(request):
    # if request.method == "POST":
    #     email = request.POST["email"]
    #     password = request.POST["password"]

    #     # Authenticate the user based on email and password
    #     user = authenticate(request, username=email, password=password)

    #     if user is not None:
    #         # User is authenticated, log them in
    #         login(request, user)
    #         return redirect("index")  # Redirect to the desired page after successful login
    #     else:
    #         # Authentication failed, show an error message or handle it as needed
    #         error_message = "Invalid email or password. Please try again."
    #         return render(request, "login.html", {"error_message": error_message})

    # return render(request, "login.html")


# def register(request):
    # if request.method == 'POST':
    #     form = RegistrationForm(request.POST)
    #     if form.is_valid():
    #         form.save()
    #         return redirect('index')
    # else:
    #     form = RegistrationForm()

    # return render(request, 'register.html')

def index(request):
    return render(request, "index.html")


def Courses(request):
    return render(request, "courses.html")



def Recommend(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Get input values
            first_name = form.cleaned_data['first_name']
            last_name = form.cleaned_data['last_name']
            sex = form.cleaned_data['sex']
            cet = form.cleaned_data['cet']
            gpa = form.cleaned_data['gpa']
            strand = form.cleaned_data['strand']

            if strand == '0':
                strand_text = 'General Academic Strand (GAS)'
            elif strand == '1':
                strand_text = 'Humanities and Social Sciences (HUMSS)'
            elif strand == '2':
                strand_text = 'Science, Technology, Engineering, and Mathematics (STEM)'
            elif strand == '3':
                strand_text = 'Technology Vocational and Livelihood (TVL)'
            elif strand == '4':
                strand_text = 'Accountancy, Business, and Management (ABM)'
            elif strand == '5':
                strand_text = 'Arts & Design (AD)'
            elif strand == '6':
                strand_text = 'Sports Track (SP)'


            # Prepare input data for prediction
            input_data = [[cet, gpa, strand]]

            # Use your model to make predictions
            model = joblib.load(r"C:\Users\acer\Desktop\thesis\myproject\model.pkl")
            decision_function_scores = model.decision_function(input_data)

            # Calculate percentages
            percentages = np.exp(decision_function_scores) / np.sum(np.exp(decision_function_scores), axis=1, keepdims=True)

            # Get the top 3 predicted courses
            top_3_courses_indices = decision_function_scores[0].argsort()[-3:][::-1]
            top_3_predicted_classes = model.classes_[top_3_courses_indices]

            course_mapping = {
                0: "BACHELOR OF ARTS IN ASIAN STUDIES",
                1: "BACHELOR OF ARTS IN BROADCASTING",
                2: "BACHELOR OF ARTS IN HISTORY",
                3: "BACHELOR OF ARTS IN ISLAMIC STUDIES",
                4: "BACHELOR OF ARTS IN JOURNALISM",
                5: "BACHELOR OF ARTS IN POLITICAL SCIENCE",
                6: "BACHELOR OF CULTURE AND ARTS EDUCATION",
                7: "BACHELOR OF EARLY CHILDHOOD EDUCATION",
                8: "BACHELOR OF ELEMENTARY EDUCATION",
                9: "BACHELOR OF PHYSICAL EDUCATION",
                10: "BACHELOR OF PUBLIC ADMINISTRATION",
                11: "BACHELOR OF SCIENCE IN ACCOUNTANCY",
                12: "BACHELOR OF SCIENCE IN ARCHITECTURE",
                13: "BACHELOR OF SCIENCE IN BIOLOGY",
                14: "BACHELOR OF SCIENCE IN CIVIL ENGINEERING",
                15: "BACHELOR OF SCIENCE IN COMMUNITY DEVELOPMENT",
                16: "BACHELOR OF SCIENCE IN COMPUTER ENGINEERING",
                17: "BACHELOR OF SCIENCE IN COMPUTER SCIENCE",
                18: "BACHELOR OF SCIENCE IN CRIMINOLOGY",
                19: "BACHELOR OF SCIENCE IN ECONOMICS",
                20: "BACHELOR OF SCIENCE IN ENVIRONMENTAL ENGINEERING",
                21: "BACHELOR OF SCIENCE IN EXERCISE AND SPORTS SCIENCES",
                22: "BACHELOR OF SCIENCE IN GEODETIC ENGINEERING",
                23: "BACHELOR OF SCIENCE IN HOME ECONOMICS",
                24: "BACHELOR OF SCIENCE IN HOSPITALITY MANAGEMENT",
                25: "BACHELOR OF SCIENCE IN INDUSTRIAL ENGINEERING",
                26: "BACHELOR OF SCIENCE IN INFORMATION TECHNOLOGY",
                27: "BACHELOR OF SCIENCE IN MECHANICAL ENGINEERING",
                28: "BACHELOR OF SCIENCE IN NURSING",
                29: "BACHELOR OF SCIENCE IN PSYCHOLOGY",
                30: "BACHELOR OF SCIENCE IN SANITARY ENGINEERING",
                31: "BACHELOR OF SCIENCE IN SOCIAL WORK",
                32: "BACHELOR OF SECONDARY EDUCATION",
                33: "BACHELOR OF SPECIAL NEED EDUCATION",
                34: "BATSILYER NG SINING SA FILIPINO",

            }

           # Calculate percentages for the top 3 courses
            top_3_percentages = percentages[0, top_3_courses_indices]

           # Determine the label for the course with the highest percentage
            highest_percentage_index = np.argmax(top_3_percentages)
            labels = [
                f"Highly Recommended! ({int(percentage * 100)}%)" if i == highest_percentage_index
                 else f"({int(percentage * 100)}%)"
                 for i, percentage in enumerate(top_3_percentages)
    ]


            course_descriptions = {

                "0": "Bachelor of Arts in Asian Studies (BA Asian Studies) is an undergraduate degree program that focuses on the interdisciplinary study of the cultures, societies, languages, history, politics, and economies of Asian countries and regions. This program provides students with a deep understanding of Asia's diverse and complex dynamics, preparing them for careers in fields such as international relations, diplomacy, business, education, and research.",
                "1": "Bachelor of Arts in Broadcasting (BA Broadcasting) is an undergraduate academic degree program designed to provide students with comprehensive knowledge and practical skills in the field of broadcasting.",
                "2": "Bachelor of Arts in History (BA History) is an undergraduate academic degree program designed to provide students with a comprehensive understanding of historical events, cultures, and societies.",
                "3": "Bachelor of Arts in Islamic Studies (BA Islamic Studies) is an undergraduate academic degree program designed to provide students with a comprehensive understanding of Islamic theology, history, culture, and related disciplines.",
                "4": "Bachelor of Arts in Journalism (BA Journalism) is an undergraduate academic degree program that focuses on preparing students for careers in journalism and media.",
                "5": "Bachelor of Arts in Political Science (BA Political Science) is an undergraduate academic degree program designed to provide students with a comprehensive understanding of political systems, government structures, political theories, and the broader field of political science.",
                "6": "Bachelor of Culture and Arts Education is an undergraduate degree program that integrates the study of culture, arts, and education. This program is designed to provide students with a comprehensive understanding of various artistic forms, cultural expressions, and their integration into the educational landscape.",
                "7": "A Bachelor of Secondary Education (B Secondary Education) is an undergraduate degree program designed to prepare individuals to become middle school and high school teachers. This program focuses on subject-specific teaching methods, educational theories, classroom management, and student assessment techniques.",
                "8": "Bachelor of Elementary Education (B Elementary Educ) is an undergraduate degree program designed to prepare individuals for careers as elementary school teachers. This program focuses on equipping future educators with the necessary knowledge, skills, and strategies to teach students in the early stages of their academic journey.",
                "9": "Bachelor of Physical Education (B Physical Educ) is an undergraduate degree program designed to prepare individuals for careers in physical education and related fields. This program focuses on providing students with a comprehensive understanding of physical activity, exercise science, sports, and health education.",
                "10": "Bachelor of Public Administration (B Public Administration) is an undergraduate degree program that focuses on preparing students for careers in public service, government agencies, and nonprofit organizations. The program combines theoretical knowledge with practical skills to equip students with the competencies needed for effective administration and management in the public sector.",
                "11": "Bachelor of Science in Accountancy (BS Accountancy) is an undergraduate degree program designed to provide students with comprehensive knowledge and skills in the field of accounting. This program prepares individuals for careers as professional accountants, auditors, and financial analysts.",
                "12": "Bachelor of Science in Architecture (BS Architecture) is an undergraduate degree program designed to provide students with a comprehensive education in the field of architecture. This program typically covers a range of subjects related to architectural theory, design, construction technology, and professional practice.",
                "13": "Bachelor of Science in Biology (BS Biology) is an undergraduate degree program that provides students with a comprehensive understanding of the biological sciences. The program typically covers a wide range of topics, including genetics, ecology, physiology, evolution, and cellular biology.",
                "14": "Bachelor of Science in Civil Engineering (BS Civil Engineering) is an undergraduate degree program that prepares students for a career in the field of civil engineering. Civil engineering involves the planning, design, construction, and maintenance of infrastructure projects, including buildings, bridges, roads, dams, and water supply systems.",
                "15": "Bachelor of Science in Community Development (BS Community Development) is an undergraduate degree program that focuses on preparing students for roles in community planning, social change, and sustainable development. This interdisciplinary program covers a range of topics related to community dynamics, social justice, and strategies for fostering positive change.",
                "16": "Bachelor of Science in Computer Engineering (BS Comp Eng) is an undergraduate degree program that combines elements of computer science and electrical engineering. This interdisciplinary program focuses on the design, development, and maintenance of computer systems and networks.",
                "17": "Bachelor of Science in Computer Science (BS Computer Science) is an undergraduate degree program that focuses on the fundamental principles of computer science, software development, and problem-solving.",
                "18": "Bachelor of Science in Criminology (BS Criminology) is an undergraduate degree program that focuses on the study of crime, criminal behavior, law enforcement, and the criminal justice system.", 
                "19": "Bachelor of Science in Economics (BS Economics) is an undergraduate degree program that provides students with a comprehensive understanding of economic principles, theories, and their applications.",
                "20": "Bachelor of Science in Environmental Engineering (BS Environmental Engineering) is an undergraduate degree program that focuses on the application of engineering principles to address environmental challenges and promote sustainability.",
                "21": "Bachelor of Science in Exercise and Sports Science (BS Exercise and Sports Science) is an undergraduate degree program that focuses on the scientific principles behind exercise, physical activity, and sports performance. This program integrates knowledge from various disciplines, including physiology, anatomy, biomechanics, psychology, and nutrition, to understand the impact of physical activity on the human body.",
                "22": "Bachelor of Science in Geodetic Engineering (BS Geodetic Engineering) is an undergraduate degree program that focuses on the principles, techniques, and applications of geodetic science and technology. Geodetic engineering involves the measurement and representation of the Earth's surface, the determination of spatial relationships, and the creation of accurate maps and geographic information systems (GIS).",
                "23": "Bachelor of Science in Home Economics (BS Home Economics) is an undergraduate degree program that focuses on the study of various aspects related to home and family life. This multidisciplinary program covers a range of subjects that contribute to the well-being and management of households.",
                "24": "Bachelor of Science in Hospitality Management (BS Hospitality Management) is an undergraduate degree program designed to prepare students for leadership roles in the dynamic and diverse field of hospitality. This program encompasses a range of subjects that focus on the management and operations of various sectors within the hospitality industry.",
                "25": "Bachelor of Science in Industrial Engineering (BS Industrial Engineering) is an undergraduate degree program that focuses on the optimization of complex processes and systems within various industries. Industrial engineering combines principles from engineering, business, and mathematics to improve efficiency, productivity, and quality.",
                "26": "Bachelor of Science in Information Technology (BS IT) is an undergraduate degree program that focuses on the principles, practices, and applications of information technology. It covers a broad spectrum of topics related to computing, technology, and the management of information systems.",
                "27": "Bachelor of Science in Mechanical Engineering (BS Mechanical Engineering) is an undergraduate degree program that focuses on the principles, design, analysis, and application of mechanical systems. Mechanical engineering is a diverse field that plays a crucial role in various industries, including manufacturing, energy, transportation, robotics, and aerospace.",
                "28": "Bachelor of Science in Nursing (BS Nursing) is an undergraduate degree program that prepares students for a career in nursing. This program provides a comprehensive education in the theory and practice of nursing, equipping students with the knowledge and skills needed to provide quality patient care.",
                "29": "Bachelor of Science in Psychology (BS Psychology) is an undergraduate degree program that provides students with a comprehensive understanding of the scientific principles and theories related to human behavior and mental processes.",
                "30": "Bachelor of Science in Sanitary Engineering (BS Sanitary Engineering) is an undergraduate degree program that focuses on the design, construction, and management of systems and facilities related to water supply, wastewater treatment, solid waste management, and environmental sanitation.",
                "31": "Bachelor of Science in Social Work (BS Social Work) is an undergraduate degree program designed to prepare students for careers in social work and related fields.",
                "32": "Bachelor of Secondary Education (B Secondary Education) is an undergraduate degree program designed to prepare individuals to become middle school and high school teachers. This program focuses on subject-specific teaching methods, educational theories, classroom management, and student assessment techniques.",
                "33": "Bachelor of Special Needs Education (B Special Needs Education) is an undergraduate degree program designed to prepare educators to work with students with diverse learning needs and disabilities. This program equips individuals with the knowledge and skills necessary to create inclusive and supportive learning environments.",
                "34": "Bachelor of Arts in Filipino (BA Filipino) is an undergraduate degree program that focuses on the study of the Filipino language, literature, and culture. This program is designed to provide students with a comprehensive understanding of the Filipino language, its literary traditions, and its cultural significance.",

            }

            top_3_predicted_courses_with_description = [
                (course_mapping[course], course_descriptions.get(str(course), "Unknown"), label)
                for course, label in zip(top_3_predicted_classes, labels)
            ]
           
            top_3_predicted_classes = [course_mapping[course_num] for course_num in top_3_predicted_classes]
            
            course_container = ""
            for course in top_3_predicted_classes:
                course_container += course + '|'

            courses = course_container[:-1].strip().split('|')

            pred_result = PredResults(
                first_name=first_name,
                last_name=last_name,
                sex=sex,
                cet=cet,
                gpa=gpa,
                strand=strand_text,
                recommended_course=courses,
                
            )
            pred_result.save()

            # for course in courses:
            #     recommended_course = RecommendedCourse(prediction_id=pred_result, course=course)    
            #     recommended_course.save()

            for course, description, percentage in top_3_predicted_courses_with_description:
                recommended_course = RecommendedCourse(prediction_id=pred_result, course=course, percentage=percentage, description=description)
                recommended_course.save()

            # Customize the message based on strand, CET, and GPA
                custom_message = f"Given your strong academic performance in {strand_text} subjects and a high GPA, This is the top recommendation. This field aligns with your educational background."

            # Pass the variables to the template
            return render(request, 'result.html', {
        'recommended_courses_with_description': top_3_predicted_courses_with_description,
        'first_name': first_name,
        'last_name': last_name,
        'sex': sex,
        'cet': cet,
        'gpa': gpa,
        'strand': strand,
        'prediction_id': pred_result.id,
        'title': 'Result',
        'custom_message': custom_message,
    })

         
    
    else:
        form = PredictionForm()
    return render(request, 'recommend.html', {'form': form, 'title': 'Recommend'})


def pdf(request, id):

    current_date = datetime.datetime.now().strftime('%B %d, %Y')
    
    # Assuming your HTML file is stored in the 'templates' directory
    template_path = 'pdf_template.html'

    # Get the required data from the database or wherever it's stored
    prediction = PredResults.objects.get(id=id)
    recommended = RecommendedCourse.objects.filter(prediction_id=prediction)

    # Render the template with context data if needed
    context = {'prediction': prediction, 'recommendeds': recommended, 'current_date': current_date}

    # Create a WeasyPrint HTML object
    html = HTML(string=render(request, template_path, context).content)

    # Generate PDF
    pdf_file = html.write_pdf()

    # Create a Django HttpResponse with the PDF content
    response = HttpResponse(pdf_file, content_type='application/pdf')

    # Optionally, you can set the Content-Disposition header to force download
    response['Content-Disposition'] = 'filename="output.pdf"'

    return response