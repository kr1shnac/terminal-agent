import sys
import tty
import termios
import select
import re
import time

def probe_terminal(sequence: str, timeout: float = 1.0) -> str:
    """Send an ANSI sequence and read the raw response from stdin."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        tty.setraw(sys.stdin.fileno())
        sys.stdout.write(sequence)
        sys.stdout.flush()
        
        response = ""
        start = time.time()
        while time.time() - start < timeout:
            r, w, x = select.select([sys.stdin], [], [], timeout - (time.time() - start))
            if not r:
                break
            char = sys.stdin.read(1)
            response += char
            
            if sequence.endswith('$p') and char == 'y':
                break
            if '11;?' in sequence and (char == '\x07' or response.endswith('\x1b\\')):
                break
                
        return response
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main():
    print("Testing Terminal Capabilities...\n")
    
    dec_query = "\x1b[?2026$p"
    dec_response = probe_terminal(dec_query, timeout=1.0)
    print(f"DEC 2026 Raw Response: {repr(dec_response)}")
    
    # Wait slightly to clear buffer if terminal was slow
    time.sleep(0.5)
    
    osc_query = "\x1b]11;?\x1b\\"
    osc_response = probe_terminal(osc_query, timeout=1.0)
    print(f"\nOSC 11 Raw Response: {repr(osc_response)}")
    
    if osc_response:
        match = re.search(r"rgb:([0-9a-fA-F/]+)", osc_response)
        if match:
            print(f"OSC 11: Background Color = {match.group(1)}")

if __name__ == "__main__":
    main()
