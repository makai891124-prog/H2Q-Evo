from django.core.management.base import BaseCommand
from h2q_project.core.models import Project, Task, UserProfile
from django.contrib.auth.models import User

class Command(BaseCommand):
    help = 'Seeds the database with initial data for testing and development.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Seeding the database...'))

        # Create a default user
        user = User.objects.create_superuser('admin', 'admin@example.com', 'password')
        self.stdout.write(self.style.SUCCESS('Created admin user.'))

        # Create a user profile
        user_profile = UserProfile.objects.create(user=user, bio='Administrator User')
        self.stdout.write(self.style.SUCCESS('Created admin user profile.'))

        # Create a sample project
        project = Project.objects.create(name='Sample Project', description='A project for demonstration purposes', owner=user_profile)
        self.stdout.write(self.style.SUCCESS('Created sample project.'))

        # Create some sample tasks
        Task.objects.create(project=project, title='Task 1', description='Description for Task 1', status='open')
        Task.objects.create(project=project, title='Task 2', description='Description for Task 2', status='in_progress')
        Task.objects.create(project=project, title='Task 3', description='Description for Task 3', status='completed')
        self.stdout.write(self.style.SUCCESS('Created sample tasks.'))

        self.stdout.write(self.style.SUCCESS('Database seeding complete.'))
