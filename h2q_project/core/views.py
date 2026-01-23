from django.shortcuts import render, get_object_or_404
from .models import Project

def project_detail(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    return render(request, 'core/project_detail.html', {'project': project})

def get_project_context(project_id):
    project = get_object_or_404(Project, pk=project_id)
    # Limit the amount of information passed to the context to reduce token usage.
    # Only include the project name and description.
    context = {
        'project_name': project.name,
        'project_description': project.description,
    }
    return context
