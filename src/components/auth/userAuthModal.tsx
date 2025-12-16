import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SignInForm } from "./userSignInForm";
import { SignUpForm } from "./userSignUpForm";
import { ForgotPasswordForm } from "./passwordResetForm";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  defaultTab?: "signin" | "signup";
}

export const AuthModal = ({ isOpen, onClose, defaultTab = "signin" }: AuthModalProps) => {
  const [activeView, setActiveView] = useState<"signin" | "signup" | "forgot">(defaultTab);

  const handleSuccess = () => {
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-2xl font-montserrat font-bold text-center">
            {activeView === "forgot" ? "Reset Password" : "Welcome to NutriSolve"}
          </DialogTitle>
        </DialogHeader>

        {activeView === "forgot" ? (
          <ForgotPasswordForm onBack={() => setActiveView("signin")} />
        ) : (
          <Tabs value={activeView} onValueChange={(v) => setActiveView(v as "signin" | "signup")} className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="signin">Sign In</TabsTrigger>
              <TabsTrigger value="signup">Sign Up</TabsTrigger>
            </TabsList>
            
            <TabsContent value="signin">
              <SignInForm 
                onSuccess={handleSuccess} 
                onForgotPassword={() => setActiveView("forgot")}
              />
            </TabsContent>
            
            <TabsContent value="signup">
              <SignUpForm onSuccess={handleSuccess} />
            </TabsContent>
          </Tabs>
        )}
      </DialogContent>
    </Dialog>
  );
};
