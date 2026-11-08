# Static page + reverse proxy to the detection API.
#
# Build:  docker build -t detector-web docker/ui
# Run:    docker run --rm -p 8080:80 -e API_HOST=api -e API_PORT=24242 detector-web
#         (API_HOST must be resolvable from this container, e.g. a compose service name)

FROM nginx:alpine

# Where the detection API lives. Override at run time with -e / compose "environment:".
ENV API_HOST=api \
    API_PORT=24242

# nginx's entrypoint renders everything in /etc/nginx/templates/*.template into
# /etc/nginx/conf.d/, filling in ${API_HOST} and ${API_PORT} from the environment.
COPY default.conf.template /etc/nginx/templates/default.conf.template
COPY index.html /usr/share/nginx/html/index.html

EXPOSE 80
