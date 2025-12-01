# Stage 1: Build the React application
FROM node:18-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . .
RUN npm run build

# Stage 2: Serve the application using Nginx
FROM nginx:1.21.3-alpine AS prod

# Copy the custom Nginx configuration that will be populated at runtime
COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=build /app/dist /usr/share/nginx/html

# The container will listen on the port defined by the PORT environment variable.
# It defaults to 8080 if the PORT variable is not set.
CMD /bin/sh -c "sed -i 's/LISTEN_PORT/'\"${PORT:-8080}\"'/' /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"
