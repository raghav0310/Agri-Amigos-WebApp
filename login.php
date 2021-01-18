<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="UTF-8" />
		<meta  name="viewport" content="width = device-width, initial-scale=1.0" />
		<meta http-equiv="X-UA-Compatible" content="ie=edge" />
		<link
			rel="stylesheet"
			href="http://use.fontawesome.com/releases/v5.8.1/css/all.css"
			integrity="sha384-50oBUHEmvpQ+1lW4y57PTFmhCaXp0ML5d60M1M7uH2+nqUivzIebhndOJK28anvf"
			crossorigin="anonymous"
		/>

		<link rel="stylesheet" href="style.css" />
		<title>Slider Sign In/Sign Up From</title>
	</head>
	<body>
	
	<div class="container" id="container">
		<div class="form-container sign-up-container">
			<form METHOD=POST action="<?PHP $_SERVER['PHP_SELF']?>">
				<h1>Create Account</h1>
				<div class="social-container">
					<a href="http:///www.google.com" class="social"><i class="fab fa-facebook-f"></i></a>
					<a href="http:///www.facebok.com" class="social"><i class="fab fa-google-plus-g"></i></a>
					<a href="http:///www.linkedIn.com" class="social"><i class="fab fa-linkedin-in"></i></a>
				</div>
				<span>or use your email for registration</span>
				<input type="text" name="sname" placeholder="Name" />
				<input type="email" name="smail" placeholder="Email" />
				<input type="password" name="spass" placeholder="Password" />
				<button><INPUT TYPE=SUBMIT NAME="BOX" ONCLICK=CLICK()>Sign Up</button>
			</form>
		</div>
		<div class="form-container sign-in-container">
			<form action="#">
				<h1>Sign in</h1>
				<div class="social-container">
					<a href="http:///www.google.com" class="social"><i class="fab fa-facebook-f"></i></a>
					<a href="http:///www.facebook.com" class="social"><i class="fab fa-google-plus-g"></i></a>
					<a href="http:///www.linkedIn.com" class="social"><i class="fab fa-linkedin-in"></i></a>
				</div>
				<span>or use your account</span>
				<input type="email" placeholder="Email" name="mail"/>
				<input type="password" placeholder="Password" name="pass"/>
				<a href="#">Forgot your password?</a>
				<button><INPUT TYPE=SUBMIT NAME="BOX1" ONCLICK=CLICK2()>Sign In</button>
			</form>
		</div>
		<div class="overlay-container">
			<div class="overlay">
				<div class="overlay-panel overlay-left">
					<h1>Welcome Back!</h1>
				<p>To keep connected with us please login with your personal info</p>
				<button class="ghost" id="signIn">Sign In</button>
			</div>
			<div class="overlay-panel overlay-right">
				<h1>Hello, Friend!</h1>
				<p>Enter your personal details and start journey with us</p>
				<button class="ghost" id="signUp">Sign Up</button>
			</div>
		</div>
	</div>
</div>

<script src="main.js"></script>
<?php
FUNCTION CLICK()
	{
	$A=$_POST["sname"];
	$B=$_POST["smail"];
	$C=$_POST["spass"];
	$servername="localhost";
	$user="root";
	$psw="";
	$dbname="sign_in";
	$conn=mysqli_connect($servername,$user,$psw,$dbname);
	if(!$conn)
	{
	die("connection failed:".mysqli_connect_error());
	}
	$sql="insert into sign values('$A','$B','$C')";
	$result=mysqli_query($conn,$sql);
	$r="select * from sign";
	$c=mysqli_query($conn,$r);
	$conn->close();
	}
IF(ISSET($_POST["BOX"]))
{
CLICK();
header("Location:http://localhost/hack/complete/index.php");
}
FUNCTION CLICK2()
	{
	$A=$_POST["mail"];
	$B=$_POST["pass"];
	$servername="localhost";
	$user="root";
	$psw="";
	$dbname="sign_in";
	$conn=mysqli_connect($servername,$user,$psw,$dbname);
	if(!$conn)
	{
	die("connection failed:".mysqli_connect_error());
	}
	$sql="select * from sign where email='$A' and password='$B'";
	$result=mysqli_query($conn,$sql);
	IF($result>0)
	{
	$r="select * from sign";
	$c=mysqli_query($conn,$r);
}
ELSE
{
	ECHO("WRONG EMAIL OR PASSWORD");
}
	$conn->close();
	}
	
IF(ISSET($_POST["BOX2"]))
{
CLICK2();
header("Location:http://localhost/hack/complete/index.php");
}
?>
</body>
</html>