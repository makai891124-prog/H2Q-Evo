from django.shortcuts import render
from django.conf import settings


def get_project_context():
    """Gets the project context for use in templates."""

    context = {
        'PROJECT_NAME': settings.PROJECT_NAME,
        'PROJECT_VERSION': settings.PROJECT_VERSION,
        'PROJECT_DESCRIPTION': settings.PROJECT_DESCRIPTION,
    }
    return context



def home(request):
    """Home page view."""
    context = get_project_context()
    return render(request, 'home.html', context)



def about(request):
    """About page view."""
    context = get_project_context()
    return render(request, 'about.html', context)
