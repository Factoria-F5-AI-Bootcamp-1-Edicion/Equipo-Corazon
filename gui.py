from tkinter import *
def sum():
    a=int(ent1.get())
    b=int(ent2.get())
    s=a+b
    output.insert(1.0,str(s))
    
win=Tk() #creating the main window and storing the window object in 'win'
win.title('Datos del paciente')
win.geometry('700x600') #setting the size of the window


text=Label(win, text='Añada los datos de analisis')
ent1 = Entry(win) 
ent2 = Entry(win) 


btn=Button(text='Add',command=sum)
output=Text(win,height=1,width=6)


text.pack()

ent1.pack()
ent2.pack()

output.pack()

btn.pack()


win.mainloop()
