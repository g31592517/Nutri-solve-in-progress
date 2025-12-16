import { Mail, Phone, MapPin, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

const Contact = () => {
  const contactInfo = [
    {
      icon: Mail,
      title: "Email",
      content: "support@nutrisolve.com",
    },
    {
      icon: Phone,
      title: "Phone",
      content: "+1 (555) 123-4567",
    },
    {
      icon: MapPin,
      title: "Location",
      content: "San Francisco, CA",
    },
  ];

  return (
    <section id="contact" className="py-20 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="font-montserrat font-bold text-4xl md:text-5xl text-foreground mb-4">
            Get in{" "}
            <span className="gradient-text">
              Touch
            </span>
          </h2>
          <p className="font-inter text-lg text-muted-foreground max-w-2xl mx-auto">
            Have questions? We'd love to hear from you. Send us a message and
            we'll respond as soon as possible.
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {contactInfo.map((info, index) => {
            const Icon = info.icon;
            return (
              <Card
                key={index}
                className="border-2 hover:border-primary/50 transition-all duration-300 hover:shadow-elegant bg-gradient-card text-center animate-fade-up"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <CardContent className="p-6">
                  <div className="w-14 h-14 rounded-lg bg-primary/10 flex items-center justify-center mx-auto mb-4">
                    <Icon className="h-7 w-7 text-primary" />
                  </div>
                  <h3 className="font-montserrat font-bold text-lg text-foreground mb-2">
                    {info.title}
                  </h3>
                  <p className="font-inter text-muted-foreground">
                    {info.content}
                  </p>
                </CardContent>
              </Card>
            );
          })}
        </div>

        <Card className="border-2 shadow-elegant bg-gradient-card max-w-2xl mx-auto mt-12">
          <CardContent className="p-8">
            <form className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label className="font-inter font-medium text-foreground mb-2 block">
                    Your Name
                  </label>
                  <Input
                    placeholder="John Doe"
                    className="border-2 focus-visible:ring-primary font-inter"
                  />
                </div>
                <div>
                  <label className="font-inter font-medium text-foreground mb-2 block">
                    Your Email
                  </label>
                  <Input
                    type="email"
                    placeholder="john@example.com"
                    className="border-2 focus-visible:ring-primary font-inter"
                  />
                </div>
              </div>
              <div>
                <label className="font-inter font-medium text-foreground mb-2 block">
                  Subject
                </label>
                <Input
                  placeholder="How can we help?"
                  className="border-2 focus-visible:ring-primary font-inter"
                />
              </div>
              <div>
                <label className="font-inter font-medium text-foreground mb-2 block">
                  Message
                </label>
                <Textarea
                  placeholder="Tell us more..."
                  rows={5}
                  className="border-2 focus-visible:ring-primary font-inter resize-none"
                />
              </div>
              <Button
                type="submit"
                size="lg"
                className="w-full bg-gradient-primary font-inter font-semibold shadow-glow hover:scale-105 transition-transform group"
              >
                Send Message
                <Send className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </section>
  );
};

export default Contact;
