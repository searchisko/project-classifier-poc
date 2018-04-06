from django.conf.urls import include, url
from django.contrib import admin

from .views import score, score_bulk, service_root

urlpatterns = [
    # Examples:
    # url(r'^$', 'project.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),
    # url(r'^health$', health),
    # url(r'^admin/', include(admin.site.urls)),
    # url(r'^scoreBulk', score_bulk)
    url(r'^scoreBulk$', score_bulk),
    url(r'^score$', score),
    url(r'^', service_root)
]



