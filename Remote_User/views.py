from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,fraud_detection,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Fraud_In_Health_Insurance(request):
    if request.method == "POST":

        if request.method == "POST":

            Fid=request.POST.get('Fid')
            ClaimID=request.POST.get('ClaimID')
            ClaimDt=request.POST.get('ClaimDt')
            Provider=request.POST.get('Provider')
            Sum_Insured=request.POST.get('Sum_Insured')
            InscClaimAmtReimbursed=request.POST.get('InscClaimAmtReimbursed')
            AttendingPhysician=request.POST.get('AttendingPhysician')
            ClmDiagnosisCode_1=request.POST.get('ClmDiagnosisCode_1')
            Claimed_Amount=request.POST.get('Claimed_Amount')
            Glucose=request.POST.get('Glucose')
            BloodPressure=request.POST.get('BloodPressure')
            SkinThickness=request.POST.get('SkinThickness')
            Insulin=request.POST.get('Insulin')
            BMI=request.POST.get('BMI')
            DiabetesPedigreeFunction=request.POST.get('DiabetesPedigreeFunction')
            Age=request.POST.get('Age')

        df = pd.read_csv('Datasets.csv')

        def apply_response(Label):
            if (Label == 0):
                return 0  # No Fraud Found
            elif (Label == 1):
                return 1  # Fraud Found

        df['results'] = df['Label'].apply(apply_response)

        cv = CountVectorizer()
        X = df['Fid']
        y = df['results']

        print("FID")
        print(X)
        print("Results")
        print(y)

        cv = CountVectorizer()
        X = cv.fit_transform(X)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Graph Neural Network-GNN")
        from sklearn.neural_network import MLPClassifier
        mlpc = MLPClassifier().fit(X_train, y_train)
        y_pred = mlpc.predict(X_test)
        testscore_mlpc = accuracy_score(y_test, y_pred)
        accuracy_score(y_test, y_pred)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('MLPClassifier', mlpc))


        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB

        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print("ACCURACY")
        print(naivebayes)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_nb))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print("ACCURACY")
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression

        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Gradient Boosting Classifier")

        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
            X_train,
            y_train)
        clfpredict = clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, clfpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, clfpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, clfpredict))
        models.append(('GradientBoostingClassifier', clf))


        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        Fid1 = [Fid]
        vector1 = cv.transform(Fid1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if (prediction == 0):
            val = 'No Fraud Found'
        elif (prediction == 1):
            val = 'Fraud Found'

        print(val)
        print(pred1)

        fraud_detection.objects.create(
        Fid=Fid,
        ClaimID=ClaimID,
        ClaimDt=ClaimDt,
        Provider=Provider,
        Sun_Insured=Sum_Insured,
        InscClaimAmtReimbursed=InscClaimAmtReimbursed,
        AttendingPhysician=AttendingPhysician,
        ClmDiagnosisCode_1=ClmDiagnosisCode_1,
        Claimed_Amount=Claimed_Amount,
        Glucose=Glucose,
        BloodPressure=BloodPressure,
        SkinThickness=SkinThickness,
        Insulin=Insulin,
        BMI=BMI,
        DiabetesPedigreeFunction=DiabetesPedigreeFunction,
        Age=Age,
        Prediction=val)

        return render(request, 'RUser/Predict_Fraud_In_Health_Insurance.html',{'objs': val})
    return render(request, 'RUser/Predict_Fraud_In_Health_Insurance.html')



