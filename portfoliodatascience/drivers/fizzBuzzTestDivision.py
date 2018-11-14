# print numbers in a range based on its division by 3 and 5 (remainder == 0)

def fizzBuzz(num):
   for i in range(1, num + 1):
       if i % 3 == 0 and i % 5 == 0:
           print(i, "FizzBuzz")
       elif i % 3 == 0:
           print(i, "Fizz")
       elif i % 5 == 0:
           print (i, "Buzz")
       else:
           print (i, "--")

fizzBuzz(100)
