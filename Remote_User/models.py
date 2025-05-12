from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):

    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)


class fraud_detection(models.Model):

    Fid= models.CharField(max_length=3000)
    ClaimID= models.CharField(max_length=3000)
    ClaimDt= models.CharField(max_length=3000)
    Provider= models.CharField(max_length=3000)
    Sun_Insured= models.CharField(max_length=3000)
    InscClaimAmtReimbursed= models.CharField(max_length=3000)
    AttendingPhysician= models.CharField(max_length=3000)
    ClmDiagnosisCode_1= models.CharField(max_length=3000)
    Claimed_Amount= models.CharField(max_length=3000)
    Glucose= models.CharField(max_length=3000)
    BloodPressure= models.CharField(max_length=3000)
    SkinThickness= models.CharField(max_length=3000)
    Insulin= models.CharField(max_length=3000)
    BMI= models.CharField(max_length=3000)
    DiabetesPedigreeFunction= models.CharField(max_length=3000)
    Age= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)



class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



