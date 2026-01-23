from django.shortcuts import render
from .models import Project


def get_project_context(request, project_id):
    try:
        project = Project.objects.get(pk=project_id)
        context = {
            'project': project
        }
        return render(request, 'project_detail.html', context)
    except Project.DoesNotExist:
        return render(request, 'project_not_found.html')
