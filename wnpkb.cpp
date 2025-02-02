#include <iostream>
#include <windows.h>

// Global variable to store the hook handle
HHOOK hKeyboardHook = nullptr;

// Low-level keyboard procedure
LRESULT CALLBACK LowLevelKeyboardProc(int nCode, WPARAM wParam, LPARAM lParam)
{
    if (nCode == HC_ACTION)
    {
        KBDLLHOOKSTRUCT* pKeyInfo = (KBDLLHOOKSTRUCT*)lParam;

        // Check if Alt key is pressed
        bool isAltPressed = (GetAsyncKeyState(VK_MENU) & 0x8000) != 0;

        // Check if the key pressed is '9'
        if (pKeyInfo->vkCode == 0x39 && isAltPressed && wParam == WM_KEYDOWN)
        {
            std::cout << "Alt + 9 pressed. Exiting..." << std::endl;
            PostQuitMessage(0); // Exit the program
        }
    }

    // Pass the event to the next hook in the chain
    return CallNextHookEx(hKeyboardHook, nCode, wParam, lParam);
}

int main()
{
    // Set the low-level keyboard hook
    hKeyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, LowLevelKeyboardProc, GetModuleHandle(nullptr), 0);
    if (hKeyboardHook == nullptr)
    {
        std::cerr << "Failed to set hook!" << std::endl;
        return 1;
    }

    std::cout << "Listening for keystrokes. Press Alt + 9 to exit." << std::endl;

    // Message loop to keep the program running
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // Unhook the keyboard hook
    UnhookWindowsHookEx(hKeyboardHook);

    return 0;
}