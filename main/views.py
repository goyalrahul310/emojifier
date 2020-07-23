from django.shortcuts import render
from django.views.generic import TemplateView
from services.emojis import app
# Create your views here.

class Index(TemplateView):
	template_name = 'index.html'

	def post(self,request):
		content = request.POST['content']
		emoji = app.predict(content)


		context = {
		"content":content,
		"emoji" : emoji
		}

		return render(request,self.template_name,context)