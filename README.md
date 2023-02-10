# Numerai example

Attempt to get a numerai model up and running with a HTTP server.

## Google cloud example

I was able to deploy the code using

```
gcloud functions deploy hello_numerai --runtime python310 --trigger-http --allow-unauthenticated --memory 1024 --timeout 300
```

and this created a function I could invoke with https://us-central1-numeraiexample.cloudfunctions.net/hello_numerai

I did have to go into the google cloud UI and set environment variables

but there is also `--set-env-vars=NUMERAI_PUBLIC_ID=X,NUMERAI_SECRET_KEY=Y` which can be invoked at CLI time to add env vars without opening the UI.

