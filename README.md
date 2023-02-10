# Numerai example

Attempt to get a numerai model up and running with a HTTP server.

## NGrok example

I run the local app with

```
NUMERAI_PUBLIC_ID=X NUMERAI_SECRET_KEY=Y python3 main.py 
```

I then run the `ngrok` command

```
# main.py listens on 4080
ngrok http 4080
```

I copy the ngrok URL `https://c706-186-72-116-169.ngrok.io` into the website info
for numerai as the compute webhook URL.

Then I run the test button in numerai UI and it seems to work.

