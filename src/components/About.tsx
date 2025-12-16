import { Check } from "lucide-react";
import { Button } from "@/components/ui/button";

const About = () => {
  const benefits = [
    "Personalized nutrition plans tailored to your unique needs",
    "AI-powered meal recommendations and recipe suggestions",
    "Track your progress with detailed analytics",
    "Access to certified nutrition experts",
    "Community support and motivation",
    "Evidence-based nutritional guidance",
  ];

  return (
    <section id="about" className="py-20">
      <div className="container mx-auto px-4">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div className="animate-fade-in">
            <div className="inline-block bg-primary/10 text-primary px-4 py-2 rounded-full mb-6 font-inter font-medium">
              About NutriSolve
            </div>
            <h2 className="font-montserrat font-bold text-4xl md:text-5xl text-foreground mb-6">
              Empowering Your{" "}
              <span className="gradient-text">
                Wellness Journey
              </span>
            </h2>
            <p className="font-inter text-lg text-muted-foreground mb-6 leading-relaxed">
              NutriSolve combines cutting-edge AI technology with proven
              nutritional science to deliver personalized guidance that fits your
              lifestyle. Whether you're managing a health condition, pursuing
              fitness goals, or simply want to eat better, we're here to support
              you every step of the way.
            </p>
            <p className="font-inter text-lg text-muted-foreground mb-8 leading-relaxed">
              Our platform adapts to your unique needs, preferences, and goals,
              making healthy eating sustainable and enjoyable.
            </p>

            <ul className="space-y-4 mb-8">
              {benefits.map((benefit, index) => (
                <li
                  key={index}
                  className="flex items-start gap-3 animate-fade-up"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Check className="h-4 w-4 text-primary" />
                  </div>
                  <span className="font-inter text-foreground/90">{benefit}</span>
                </li>
              ))}
            </ul>

            <Button
              size="lg"
              className="bg-gradient-primary font-inter font-semibold shadow-glow hover:scale-105 transition-transform"
            >
              Discover More
            </Button>
          </div>

          <div className="relative animate-scale-in">
            <div className="aspect-square rounded-2xl bg-gradient-primary/10 p-8 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-64 h-64 bg-primary/20 rounded-full blur-3xl" />
              <div className="absolute bottom-0 left-0 w-64 h-64 bg-accent/20 rounded-full blur-3xl" />
              <div className="relative z-10 h-full flex items-center justify-center">
                <div className="text-center">
                  <div className="w-32 h-32 mx-auto mb-6 rounded-full bg-gradient-primary flex items-center justify-center shadow-glow">
                    <span className="text-6xl">🥗</span>
                  </div>
                  <p className="font-montserrat font-bold text-2xl text-foreground">
                    Your Health,
                    <br />
                    <span className="text-primary">Our Mission</span>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
