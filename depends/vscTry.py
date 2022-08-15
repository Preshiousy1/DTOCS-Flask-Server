from time import strftime
import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import tkcalendar
from tkcalendar import *
from tkinter.filedialog import askopenfile, askopenfilename
import tkinter.filedialog as tkFileDialog
from datetime import timedelta
from datetime import *
import datetime as dt
import re
import csv
import pandas as pd

current_date = dt.date.today()
date_format = "%m%d%Y"

# def sched_start_call():
#     top = tk.Tk()

#     ttk.Label(top, text='Choose date').pack(padx=10, pady=10)

#     cal = DateEntry(top, width=12, background='darkblue',
#                     foreground='white', borderwidth=2, year=current_date.year)
#     cal.pack(padx=10, pady=10)

def enter_data():
 #   accepted = accept_var.get()
    
        # User info
    schedstart = sched_start_entry.get()
    horizon = horizon_entry.get()
    #horizon_int = int(horizon)
    
    if schedstart and horizon:
        schedend = schedend.get()
        age = age_spinbox.get()
        nationality = nationality_combobox.get()
        
        # Course info
        # registration_status = reg_status_var.get()
        # numcourses = numcourses_spinbox.get()
        # numsemesters = numsemesters_spinbox.get()
        
        print("Schedule Start: ", schedstart, "Horizon(in days): ", horizon)
        print("Title: ", title, "Age: ", age, "Nationality: ", nationality)
        print("# Courses: ", numcourses, "# Semesters: ", numsemesters)
        print("Registration status", registration_status)
        print("------------------------------------------")
    else:
        tkinter.messagebox.showwarning(title="Error", message="Schedule start and horizon entries are required.")


window = tkinter.Tk()
window.title("Data Entry Form")

frame = tkinter.Frame(window)
frame.pack()

# Saving User Info
user_info_frame =tkinter.LabelFrame(frame, text="Schedule and Horizon Information")
user_info_frame.grid(row= 0, column=0, padx=20, pady=10)



# sched_start_entry = tkinter.Entry(user_info_frame)
sched_start_entry = tkcalendar.DateEntry( user_info_frame)
sched_start_entry.grid(row=1, column=0)
schedstart_date=sched_start_entry.get_date()

horizon_entry = tkinter.Entry(user_info_frame)
horizon_entry.grid(row=1, column=1)
horizon_entry.insert(0,"0")


def get_sched_date():
    
    global sched_sd
    global sched_sm
    global sched_sy
    global sched_date
    global schedend
    global horizon_int
    global schedend_label1
   
    try:
        sched_date = dt.datetime.strftime(schedstart_date, "%Y-%m-%d")
        sched_sd = dt.datetime.strftime(schedstart_date,"%d")
        sched_sm = dt.datetime.strftime(schedstart_date,"%m")
        sched_sy = dt.datetime.strftime(schedstart_date,"%Y")
        schedule_s = dt.datetime(int(sched_sy), int(sched_sm), int(sched_sd))
        horizon_num = float(horizon_entry.get())
        horizon_int = int(horizon_num)
        schedend = schedule_s + timedelta(days=horizon_int)
        schedend_label1= tkinter.Label(user_info_frame, text=dt.datetime.strftime(schedend, "%Y-%m-%d"))
        schedend_label1.grid(row=1, column=2)
        schedend_label1.config(bg= "white", fg= "black")
        
    except ValueError:
        tkinter.messagebox.showerror("Error. Not a valid entry ")
        return None
    
    
    schedend_label1.update()
    
    return schedend

        


sched_start_label = tkinter.Label(user_info_frame, text="Schedule Start")
sched_start_label.grid(row=0, column=0)
horizon_label = tkinter.Label(user_info_frame, text="Horizon (in days)")
horizon_label.grid(row=0, column=1)
print(horizon_entry.get()) 


# get_sched_date()

def get_sched_end():
    print(schedend)

schedend_label = tkinter.Label(user_info_frame, text="Schedule End")
schedend_label.grid(row=0, column=2)

#schedend_label1.config(text=schedend)



sched_button=tkinter.Button(user_info_frame, text='Update Schedule End', command=get_sched_date).grid(row=1, column=3)


csv_label = tkinter.Label(user_info_frame, text="Input CSV")
csv_label.grid(row=2, column=0)

v=tkinter.StringVar()

def csv_getname():
    csv_filename=tkFileDialog.askopenfilename(initialdir="/", title="Select CSV", filetypes=(("CSV files", "*.csv"), ("All files","*.*")))
    csv_label1.config(text=csv_filename)  
    get_csv(csv_filename)


csv_button=tkinter.Button(user_info_frame, text='Browse Data Set', command=csv_getname).grid(row=3, column=0)

csv_label1 = tkinter.Label(user_info_frame, text='')
csv_label1.grid(row=4, column=1)



# def get_csv(csv_filename):
#     global csv_in
#     csv_in = pd.read_csv(csv_filename)
#     column_combobox.config(values=csv_in.columns.to_list())


list_of_column_names = [] #list(csv_in.columns)
 
datatype_list =[
    str,
    float
]  

def add_datatype(e):
    if column_combobox.get()!='':
        datatype_combobox.config(values=datatype_list)
        datatype_combobox.current(0)


def column_search(event):
    value=event.widget.get()
    if value =='':
        column_combobox['value'] = list_of_column_names  
    else:
        data =[]
        
        for item in list:
            if value.lower() in item.lower():
                data.append(item)
            column_combobox['values'] = data
            

def add_data_tree():
    if column_combobox.get() and datatype_combobox.get():
        print(column_combobox.get()+" and "+datatype_combobox.get())
    else:
        print("Please pick column name and datatype")
    

def Load_excel_data():
    """If the file selected is valid this will load the file into the Treeview"""
    #file_path = label_file["text"]
    file_path=csv_filename["text"]["text"]
    try:
        excel_filename = r"{}".format(file_path)
        if excel_filename[-4:] == ".csv":
            df = pd.read_csv(excel_filename)
        else:
            df = pd.read_excel(excel_filename)

    except ValueError:
        tk.messagebox.showerror("Information", "The file you have chosen is invalid")
        return None
    except FileNotFoundError:
        tk.messagebox.showerror("Information", f"No such file as {file_path}")
        return None

    clear_data()
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column) # let the column heading = column name
        tv1.insert("","end", values=df.columns.to_list)
    df_rows = df.to_numpy().tolist() # turns the dataframe into a list of lists
    for row in df_rows:
        tv1.insert("", "end", values=row) # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert
    return None


def clear_data():
    tv1.delete(*tv1.get_children())
    return None

def get_csv(csv_filename):
    global csv_in
    csv_in = pd.read_csv(csv_filename)
    print('found the csv file')
    print(csv_in.columns)
    column_combobox.config(values=csv_in.columns.to_list())
    tv1.config(columns=csv_in.columns.to_list(), show='headings')
    test_frame = csv_in.iloc[:,:]
    for item in csv_in.columns.to_list():
        tv1.insert('', 'end', values=item)

            
column_label = tkinter.Label(user_info_frame, text="Select Columns")
column_combobox = ttk.Combobox(user_info_frame, values= list_of_column_names)
column_combobox.bind("<<ComboboxSelected>>", add_datatype)
# column_listbox = tkinter.Listbox(user_info_frame, values= list_of_column_names)
column_label.grid(row=2, column=1)
column_combobox.grid(row=3, column=1)
# column_listbox.grid(row=3, column=1)

datatype_label = tkinter.Label(user_info_frame, text="Select Datatype")
datatype_combobox = ttk.Combobox(user_info_frame, values= datatype_list)
datatype_combobox.bind("<<ComboboxSelected>>", get_csv)
#datatype_combobox.bind("<<ComboboxSelected>>", add_data_tree)
# column_listbox = tkinter.Listbox(user_info_frame, values= list_of_column_names)
datatype_label.grid(row=2, column=2)
datatype_combobox.grid(row=3, column=2 )

for widget in user_info_frame.winfo_children():
    widget.grid_configure(padx=10, pady=5)

# Saving Course Info
courses_frame = tkinter.LabelFrame(frame)
courses_frame.grid(row=1, column=0, sticky="news", padx=20, pady=10)

tv1=ttk.Treeview(courses_frame)
tv1.place(relheight=1,relwidth=1)

treescrolly=tkinter.Scrollbar(courses_frame, orient="vertical", command=tv1.yview)
treescrollx=tkinter.Scrollbar(courses_frame, orient="horizontal", command=tv1.xview)
tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
treescrollx.pack(side="bottom", fill="x")
treescrolly.pack(side="right", fill="y")




# Button
button = tkinter.Button(frame, text="Enter data", command= enter_data)
button.grid(row=3, column=0, sticky="news", padx=20, pady=10)
 
window.mainloop()