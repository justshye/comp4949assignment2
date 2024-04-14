import pickle

import pandas as pd
from django.shortcuts import render, HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView


def homePageView(request):
    # return request object and specify page.
    return render(request, 'home.html', {
        'mynumbers': [0, 1, 2, 3, ]})


def homePost(request):
    # Use request object to extract choice.

    blueKills = -999
    blueDragons = -999
    redKills = -999
    redDragons = -999

    try:
        # Extract value from request object by control name.
        blueDragonChoice = request.POST['blueDragons']
        redDragonChoice = request.POST['redDragons']
        blueKills = request.POST['blueKills']
        redKills = request.POST['redKills']

        # Crude debugging effort.
        print("*** Blue Dragons Slain: " + str(blueDragonChoice))
        blueDragons = int(blueDragonChoice)
        print("*** Red Dragons Slain: " + str(redDragonChoice))
        redDragons = int(redDragonChoice)
    # Enters 'except' block if integer cannot be created.
    except:
        return render(request, 'home.html', {
            'errorMessage': '*** The data submitted is invalid. Please try again.',
            'mynumbers': [0, 1, 2, 3, ]})
    else:
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('results', kwargs={'blueDragons': blueDragons, 'blueKills': blueKills,
                                                               'redDragons': redDragons, 'redKills': redKills}, ))


def results(request, blueDragons, blueKills, redDragons, redKills):
    print("*** Inside results()")
    # Load the saved model
    with open('./model_pkl', 'rb') as f:
        loadedModel = pickle.load(f)

    # Create a single prediction DataFrame
    singleSampleDf = pd.DataFrame({
        'blueTeamTotalKills': [blueKills],
        'blueTeamDragonKills': [blueDragons],
        'redTeamDragonKills': [redDragons],
        'redTeamTotalKills': [redKills]
    })

    singlePrediction = loadedModel.predict(singleSampleDf)

    print("Single prediction: " + str(singlePrediction))

    return render(request, 'results.html', {
        'blueDragons': blueDragons,
        'blueKills': blueKills,
        'redDragons': redDragons,
        'redKills': redKills,
        'prediction': singlePrediction
    })



def aboutPageView(request):
    # return request object and specify page.
    return render(request, 'about.html')
