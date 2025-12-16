import { Star } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

const Testimonials = () => {
  const testimonials = [
    {
      name: "Sarah Johnson",
      role: "Fitness Enthusiast",
      content:
        "NutriSolve transformed how I approach nutrition. The AI recommendations are spot-on, and I've never felt healthier!",
      rating: 5,
      avatar: "👩‍💼",
    },
    {
      name: "Michael Chen",
      role: "Busy Professional",
      content:
        "As someone with a hectic schedule, having personalized meal plans at my fingertips has been a game-changer. Highly recommend!",
      rating: 5,
      avatar: "👨‍💼",
    },
    {
      name: "Emily Rodriguez",
      role: "Wellness Coach",
      content:
        "I recommend NutriSolve to all my clients. The combination of AI technology and expert guidance is unmatched.",
      rating: 5,
      avatar: "👩‍⚕️",
    },
  ];

  return (
    <section id="testimonials" className="py-20 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="font-montserrat font-bold text-4xl md:text-5xl text-foreground mb-4">
            What Our Users{" "}
            <span className="gradient-text">
              Are Saying
            </span>
          </h2>
          <p className="font-inter text-lg text-muted-foreground max-w-2xl mx-auto">
            Join thousands of satisfied users on their journey to better health
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <Card
              key={index}
              className="border-2 hover:border-primary/50 transition-all duration-300 hover:shadow-elegant bg-gradient-card animate-fade-up"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <CardContent className="p-6">
                <div className="flex items-center gap-1 mb-4">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star
                      key={i}
                      className="h-5 w-5 fill-accent text-accent"
                    />
                  ))}
                </div>
                <p className="font-inter text-foreground/90 mb-6 leading-relaxed">
                  "{testimonial.content}"
                </p>
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center text-2xl">
                    {testimonial.avatar}
                  </div>
                  <div>
                    <div className="font-montserrat font-semibold text-foreground">
                      {testimonial.name}
                    </div>
                    <div className="font-inter text-sm text-muted-foreground">
                      {testimonial.role}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Testimonials;
