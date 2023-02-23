from simple_term_menu import TerminalMenu
print("¡Hola! Introduce los datos del nuevo paciente\n")

terminal_menu = TerminalMenu(["entry 1", "entry 2", "entry 3"])
choice_index = terminal_menu.show()

print(choice_index)