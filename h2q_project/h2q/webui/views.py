from django.shortcuts import render
from h2q.core.models import Project


def get_project_context(project_slug):
    project = Project.objects.get(slug=project_slug)
    # Directly pass the Project object to the context, instead of its attributes
    context = {
        'project': project
    }
    return context


def project_home(request, project_slug):
    context = get_project_context(project_slug)
    return render(request, 'project_home.html', context)


def project_about(request, project_slug):
    context = get_project_context(project_slug)
    return render(request, 'project_about.html', context)


def project_dashboard(request, project_slug):
    context = get_project_context(project_slug)
    return render(request, 'project_dashboard.html', context)
