import io

buf = io.BytesIO()
buf.write(b"Hello, World!")
buf.seek(0)
print(buf.read())
