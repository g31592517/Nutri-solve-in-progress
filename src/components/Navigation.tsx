import { useState, useEffect } from "react";
import { Menu, X, Leaf, User, Flame } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from "@/contexts/AuthContext";
import { useGamification } from "@/contexts/GamificationContext";
import { useToast } from "@/hooks/use-toast";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { UserDashboard } from "@/components/profile/UserDashboard";

interface NavigationProps {
  onOpenAuth: (tab?: "signin" | "signup") => void;
}

const Navigation = ({ onOpenAuth }: NavigationProps) => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);
  const { user, logout } = useAuth();
  const { gamification } = useGamification();
  const { toast } = useToast();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  const handleSignOut = () => {
    logout();
    toast({
      title: "Signed out",
      description: "You've been signed out successfully.",
    });
  };

  const navLinks = [
    { href: "#home", label: "Home" },
    { href: "#features", label: "Features" },
    { href: "#about", label: "About" },
    { href: "#specialists", label: "Specialists" },
    { href: "#testimonials", label: "Testimonials" },
    { href: "#community", label: "Community" },
    { href: "#contact", label: "Contact" },
  ];

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 border-b-2 ${
        isScrolled
          ? "bg-background/95 backdrop-blur-md shadow-elegant border-primary/20"
          : "bg-background/80 backdrop-blur-sm border-transparent"
      }`}
    >
      <nav className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <a
            href="#home"
            className="flex items-center gap-2 font-montserrat font-bold text-2xl text-foreground hover:text-primary transition-colors"
          >
            <Leaf className="h-8 w-8 text-primary" />
            <span>NutriSolve</span>
          </a>

          {/* Desktop Navigation */}
          <ul className="hidden md:flex items-center gap-8">
            {navLinks.map((link) => (
              <li key={link.href}>
                <a
                  href={link.href}
                  className="font-inter text-foreground/80 hover:text-primary transition-colors font-medium relative group"
                >
                  {link.label}
                  <span className="absolute left-0 bottom-0 w-0 h-0.5 bg-primary group-hover:w-full transition-all duration-300" />
                </a>
              </li>
            ))}
          </ul>

          <div className="hidden md:flex items-center gap-4">
            {user ? (
              <>
                {gamification.streaks.currentStreak > 0 && (
                  <div className="flex items-center gap-1 px-3 py-1 bg-primary/10 rounded-full text-sm font-medium">
                    <Flame className="h-4 w-4 text-orange-500" />
                    <span>{gamification.streaks.currentStreak}</span>
                  </div>
                )}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="rounded-full">
                      <User className="h-5 w-5" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-56">
                    <DropdownMenuLabel>
                      <div>
                        <p className="font-semibold">{user.username}</p>
                        <p className="text-xs text-muted-foreground font-normal">{user.email}</p>
                      </div>
                    </DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => setShowDashboard(true)}>
                      <User className="mr-2 h-4 w-4" />
                      My Dashboard
                    </DropdownMenuItem>
                    <DropdownMenuItem>
                      <Flame className="mr-2 h-4 w-4" />
                      {gamification.points} Points
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={handleSignOut}>
                      Sign Out
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </>
            ) : (
              <>
                <Button variant="ghost" size="lg" className="font-inter" onClick={() => onOpenAuth("signin")}>
                  Sign In
                </Button>
                <Button
                  size="lg"
                  className="bg-gradient-primary font-inter font-semibold shadow-glow hover:scale-105 transition-transform"
                  onClick={() => onOpenAuth("signup")}
                >
                  Get Started
                </Button>
              </>
            )}
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden p-2 text-foreground hover:text-primary transition-colors"
            aria-label="Toggle menu"
          >
            {isMobileMenuOpen ? (
              <X className="h-6 w-6" />
            ) : (
              <Menu className="h-6 w-6" />
            )}
          </button>
        </div>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <div className="md:hidden mt-4 pb-4 animate-fade-in">
            <ul className="flex flex-col gap-4">
              {navLinks.map((link) => (
                <li key={link.href}>
                  <a
                    href={link.href}
                    onClick={() => setIsMobileMenuOpen(false)}
                    className="block font-inter text-foreground/80 hover:text-primary transition-colors font-medium py-2"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
              <li className="flex flex-col gap-2 pt-2">
                {user ? (
                  <>
                    <Button variant="ghost" className="w-full font-inter" onClick={() => { setShowDashboard(true); setIsMobileMenuOpen(false); }}>
                      <User className="mr-2 h-4 w-4" />
                      My Dashboard
                    </Button>
                    <Button variant="ghost" className="w-full font-inter" onClick={handleSignOut}>
                      Sign Out
                    </Button>
                  </>
                ) : (
                  <>
                    <Button variant="ghost" className="w-full font-inter" onClick={() => onOpenAuth("signin")}>
                      Sign In
                    </Button>
                    <Button className="w-full bg-gradient-primary font-inter font-semibold" onClick={() => onOpenAuth("signup")}>
                      Get Started
                    </Button>
                  </>
                )}
              </li>
            </ul>
          </div>
        )}
      </nav>

      {/* Dashboard Modal */}
      <Dialog open={showDashboard} onOpenChange={setShowDashboard}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <UserDashboard onEditProfile={() => {}} />
        </DialogContent>
      </Dialog>
    </header>
  );
};

export default Navigation;
