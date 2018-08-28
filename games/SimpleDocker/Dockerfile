FROM python:3.6

# On Server
# ENV http_proxy http://10.61.3.150:8088
# ENV https_proxy http://10.61.3.150:8088
# ENV no_proxy 127.0.0.1,/var/run/docker.sock

ADD ./examples/docker-agent /agent

# @TODO to be replaced with `pip install pommerman`
ADD . /pommerman
RUN cd /pommerman && pip install .
# end @TODO

EXPOSE 10080

ENV NAME Agent

# Run app.py when the container launches
WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["run.py"]
