<!DOCTYPE html>
<html lang="en">
<head>
<title>Artistry</title>
<meta charset="utf-8">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="description" content="Wish shop project">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" type="text/css" href="styles/bootstrap4/bootstrap.min.css">
<link href="plugins/font-awesome-4.7.0/css/font-awesome.min.css" rel="stylesheet" type="text/css">
<link rel="stylesheet" type="text/css" href="plugins/OwlCarousel2-2.2.1/owl.carousel.css">
<link rel="stylesheet" type="text/css" href="plugins/OwlCarousel2-2.2.1/owl.theme.default.css">
<link rel="stylesheet" type="text/css" href="plugins/OwlCarousel2-2.2.1/animate.css">
<link href="plugins/colorbox/colorbox.css" rel="stylesheet" type="text/css">
<link rel="stylesheet" type="text/css" href="styles/main_styles.css">
<link rel="stylesheet" type="text/css" href="styles/responsive.css">
<style type="text/css">
body,td,th {
	font-family: "Open Sans", sans-serif;
}
</style>
</head>
<body >

<div class="super_container">
	
	<!-- Header -->

	<header class="header">
		<div class="header_inner d-flex flex-row align-items-center justify-content-start">
			<div class="logo">
				<p><img src="images/Background/ART.png" height=80 width=250></p>
            </div>
			<nav class="main_nav">
				<ul>
					<li><a href="http://localhost/hack/complete/index.php">home</a></li>
					<li><a href="#handi">Handicrafts</a></li>
					<li><a href="#jewel">Jewellery</a></li>
					<li><a href="#paint">Paintings</a></li>
					<li><a href="contact.php">Contact</a></li>
				</ul>
			</nav>
			<div class="header_content ml-auto">
				<div class="search header_search">
					<form action="#">
						<input type="search" class="search_input" required="required">
						<button type="submit" id="search_button" class="search_button"><img src="images/magnifying-glass.svg" alt=""></button>
					</form>
				</div>
				<div class="shopping">
					<!-- Cart -->
					<a href="#">
						<div class="cart">
							<img src="images/shopping-bag.svg" alt="">
							<div class="cart_num_container">
								<div class="cart_num_inner">
									<div class="cart_num">1</div>
								</div>
							</div>
						</div>
					</a>
					<!-- Star -->
					<a href="#">
						<div class="star">
							<img src="images/star.svg" alt="">
							<div class="star_num_container">
								<div class="star_num_inner">
									<div class="star_num">0</div>
								</div>
							</div>
						</div>
					</a>
					<!-- Avatar -->
					<a href="cart.php">
						<div class="avatar">
							<img src="images/avatar.svg" alt="">
						</div>
					</a>
				</div>
			</div>

			<div class="burger_container d-flex flex-column align-items-center justify-content-around menu_mm"><div></div><div></div><div></div></div>
		</div>
	</header>

	<!-- Menu -->

	<div class="menu d-flex flex-column align-items-end justify-content-start text-right menu_mm trans_400">
		<div class="menu_close_container"><div class="menu_close"><div></div><div></div></div></div>
		<div class="logo menu_mm"><img src="images/Background/ART.png" height=80 width=250></div>
		<div class="search">
			<form action="#">
				<input type="search" class="search_input menu_mm" required="required">
				<button type="submit" id="search_button_menu" class="search_button menu_mm"><img class="menu_mm" src="images/magnifying-glass.svg" alt=""></button>
			</form>
		</div>
		<nav class="menu_nav">
			<ul class="menu_mm">
				<li class="menu_mm"><a href="http://localhost/hack/complete/index.php">Home</a></li>
				<li class="menu_mm"><a href="#handi">Handicraft</a></li>
				<li class="menu_mm"><a href="#jewel">Jewellery</a></li>
				<li class="menu_mm"><a href="#paint">Painting</a></li>
				<li class="menu_mm"><a href="contact.php">Contact</a></li>
			</ul>
		</nav>
	</div>

	<!-- Home -->

	<div class="home">
		
		<!-- Home Slider -->

		<div class="home_slider_container">
			<div class="owl-carousel owl-theme home_slider">
				
				<!-- Home Slider Item -->
				<div class="owl-item">
					<div class="home_slider_background" style="background-image:url(images/wallpaper/w1.jpg)"></div>
					<div class="home_slider_content">
						<div class="home_slider_content_inner">
							<div class="home_slider_subtitle">Artistry</div>
							<div class="home_slider_title">The Power Of ARTS</div>
						</div>	
					</div>
				</div>

				<!-- Home Slider Item -->
				<div class="owl-item">
					<div class="home_slider_background" style="background-image:url(images/wallpaper/w2.jpg)"></div>
					<div class="home_slider_content">
						<div class="home_slider_content_inner">
							<div class="home_slider_subtitle">Exclusive</div>
							<div class="home_slider_title">Handpicked Products</div>
						</div>	
					</div>
				</div>

				<!-- Home Slider Item -->
				<div class="owl-item">
					<div class="home_slider_background" style="background-image:url(images/wallpaper/w3.jpg)"></div>
					<div class="home_slider_content">
						<div class="home_slider_content_inner">
							<div class="home_slider_subtitle">Rarest Of Rare</div>
							<div class="home_slider_title">New Collection</div>
						</div>	
					</div>
				</div>

			</div>
			
			<!-- Home Slider Nav -->

			<div class="home_slider_next d-flex flex-column align-items-center justify-content-center"><img src="images/arrow_r.png" alt=""></div>

			<!-- Home Slider Dots -->

			<div class="home_slider_dots_container">
				<div class="container">
					<div class="row">
						<div class="col">
							<div class="home_slider_dots">
								<ul id="home_slider_custom_dots" class="home_slider_custom_dots">
									<li class="home_slider_custom_dot active"><div></div></li>
									<li class="home_slider_custom_dot"><div></div></li>
									<li class="home_slider_custom_dot"><div></div></li>
								</ul>
							</div>
						</div>
					</div>
				</div>		
			</div>
		</div>
	</div>

	<!-- Promo -->
<form method=POST action="<?PHP $_SERVER['PHP_SELF']?>">
	<div class="promo">
		<div class="container">
			<div class="row">
				<div class="col">
					<div class="section_title_container text-center">
						<div class="section_subtitle">only the best</div>
						<div class="section_title">Art Piece</div>
					</div>
				</div>
			</div>
			<div class="row promo_container">

				<!-- Promo Item -->
				<div class="col-lg-4 promo_col">
					<div class="promo_item">
						<div class="promo_image">
							<center>
							<img src="images/handicraft.jpg" alt="" height="300">
							</center>
							<div class="promo_content promo_content_1">
								<div class="promo_title">-30% off</div>
								<div class="promo_subtitle">Handicraft</div>
							</div>
						</div>
						<div class="promo_link"><input type=submit onClick="CLICK1()" name="box1" value="Shop NoW"></div>
					</div>
				</div>

				<!-- Promo Item -->
				<div class="col-lg-4 promo_col">
					<div class="promo_item">
						<div class="promo_image">
							<center>
							<img src="images/palm-tree.jpg" alt="" height="300">
							</center>
							<div class="promo_content promo_content_2">
							  <div class="promo_title" >30% off</div>
								<div class="promo_subtitle">Home Decor</div>
							</div>
						</div>
						<div class="promo_link" ><input type=submit onClick="CLICK2()" name="box2" value="Shop NoW"></div>
					</div>
				</div>

				<!-- Promo Item -->
				<div class="col-lg-4 promo_col">
					<div class="promo_item">
						<div class="promo_image">
							<center>
							<img src="images/Handicraft-PNG-Picture.png" alt="" height="300">
							</center>
							<div class="promo_content promo_content_3">
							  <div class="promo_title">-25% off</div>
								<div class="promo_subtitle">on Sculpture</div>
							</div>
						</div>
						<div class="promo_link"><input name="box3" type="submit" onclick="CLICK3()" value="Shop Now"></div>
					</div>
				</div>

			</div>
		</div>
	</div>
</form>
	<!-- New Arrivals -->
	<a name="handi"><font color="white">yes</font></a>
	<div class="arrivals">
		<div class="container">
			<div class="row">
				<div class="col">
					<div class="section_title_container text-center">
						<div class="section_subtitle">only the best</div>
						<div class="section_title">HANDICRAFTS</div>
					</div>
				</div>
			</div>
			<div class="row products_container">

				<!-- Promo Item -->
				<div class="col-lg-4 promo_col">
					<div class="promo_item">
						<div class="promo_image">
							<img src="images/rajasthan/rajashandi.jpg" alt="" height="250">
							<div class="promo_content promo_content_2">
								<div class="promo_title">rAREST oF</div>
								<div class="promo_subtitle">rAJASTHANI</div>
							</div>
						</div>
						<div class="promo_link"><a href="handicraftsofrajasthan.html">Shop Now</a></div>
					</div>
				</div>
				
				<!-- Promo Item -->
				<div class="col-lg-4 promo_col">
					<div class="promo_item">
						<div class="promo_image">
							<img src="images/palm-tree.jpg" alt="" height="250">
							<div class="promo_content promo_content_2">
							  <div class="promo_title">gLORY oF</div>
								<div class="promo_subtitle">gUJARAT</div>
							</div>
						</div>
						<div class="promo_link"><a href="handicraftsofgujarat.html">Shop Now</a></div>
					</div>
				</div>
			
		
				<!-- Promo Item -->
				<div class="col-lg-4 promo_col">
					<div class="promo_item">
						<div class="promo_image">
							<img src="images/Handicraft-PNG-Picture.png" alt="" height="250">
							<div class="promo_content promo_content_2">
								<div class="promo_title">kINGS oF </div>
								<div class="promo_subtitle">kASHMIRI</div>
							</div>
						</div>
						<div class="promo_link"><a href="handicraftsofjammu.html">Shop Now</a></div>
					</div>
				</div>
				
				<div class="row con">
				<!-- Promo Item -->
				<div class="promo_col col-lg-5 offset-lg-2">
					<div class="promo_item">
						<div class="promo_image">
							<img src="images/nagaland/handicraft.jpg" alt="" width="390" height="250">
							<div class="promo_content promo_content_2">
								<div class="promo_title">nOSTALGIA oF</div>
								<div class="promo_subtitle">nAGALAND</div>
							</div>
						</div>
						<div class="promo_link"><a href="handicraftsofnagaland.html">Shop Now</a></div>
					</div>
				</div>			

				<!-- Promo Item -->
				<div class="col-lg-4 promo_col offset-lg-0">
					<div class="promo_item">
						<div class="promo_image">
							<img src="images/karnataka/236800_2.jpg" alt="" width="550" height="250">
							<div class="promo_content promo_content_2">
								<div class="promo_title">kINECTICS oF </div>
								<div class="promo_subtitle">kERALA</div>
							</div>
						</div>
						<div class="promo_link"><a href="handicraftsofkerela.html">Shop Now</a></div>
					</div>
				</div>
			</div>
		</div>
	</div>
		</div>

<!-- New Arrivals -->
	<a name="jewel"><font color="white">yes</font></a>
	<div class="arrivals">
		<div class="container">
			<div class="row">
				<div class="col">
					<div class="section_title_container text-center">
						<div class="section_subtitle">only the best</div>
						<div class="section_title">JEWELLERY</div>
					</div>
				</div>
			</div>
		  <div class="row products_container">

				<!-- Promo Item -->
				<div class="col-lg-4 promo_col">
					<div class="promo_item">
						<div class="promo_image">
							<center><img src="images/karnataka/jcp.jpg" alt="" height="250"></center>
							<div class="promo_content promo_content_2">
								<div class="promo_title">kINETICS oF</div>
								<div class="promo_subtitle">kERELA</div>
							</div>
						</div>
						<div class="promo_link"><a href="jewelsofkerela.html">Shop Now</a></div>
					</div>
				</div>
				<!-- Promo Item -->
				<div class="col-lg-4 promo_col">
					<div class="promo_item">
						<div class="promo_image">
						 <center><img src="images/rajasthan/rajasjewel.jpg" alt="" height="250"></center>
							<div class="promo_content promo_content_2">
								<div class="promo_title">rAREST oF</div>
								<div class="promo_subtitle">rAJASTHANI</div>
							</div>
						</div>
						<div class="promo_link"><a href="jewelsofrajasthan.html">Shop Now</a></div>
					</div>
				</div>
				<!-- Promo Item -->
				<div class="col-lg-4 promo_col">
					<div class="promo_item">
						<div class="promo_image">
							<center><img src="images/nagaland/713VE-aXS-L._UY625_.jpg" alt="" height="250"></center>
							<div class="promo_content promo_content_2">
								<div class="promo_title">nOSTALGIA oF</div>
								<div class="promo_subtitle">nAGALAND</div>
							</div>
						</div>
						<div class="promo_link"><a href="jewelsofnagaland.html">Shop Now</a></div>
					</div>
				</div>
			</div>
			<div class="row con	">
				<!-- Promo Item -->
				<div class="promo_col col-lg-5 offset-lg-2">
					<div class="promo_item">
						<div class="promo_image col-lg-12">
							<center><img  src="images/gujarat/gujrajewel.jfif" alt="" height="250" ></center>
							<div class="promo_content promo_content_2">
								<div class="promo_title">gLORY oF</div>
								<div class="promo_subtitle">gUJARAT</div>
							</div>
						</div>
						<div class="promo_link col-lg-12"><a href="jewelsofgujarat.html">Shop Now</a></div>
					</div>
				</div>			

				<!-- Promo Item -->
				<div class="promo_col offset-lg-0 col-lg-3">
					<div class="promo_item">
						<div class="promo_image">
							<center><img src="images/kashmir/8058f02e7dbce6d5bc1f769d92442921.jpg" alt="" height="250"></center>
							<div class="promo_content promo_content_2">
								<div class="promo_title">kINGS oF </div>
								<div class="promo_subtitle">kASHMIRI</div>
							</div>
						</div>
						<div class="promo_link"><a href="jewelsofkashmir.html">Shop Now</a></div>
					</div>
				</div>

				
			  </div>
		</div>
	</div>
	<!-- New Arrivals -->
	<a name="paint"><font color="white">yes</font></a>
	<div class="arrivals">
		<div class="container">
			<div class="row">
				<div class="col">
					<div class="section_title_container text-center">
						<div class="section_subtitle">only the best</div>
						<div class="section_title">PAINTINGS</div>
					</div>
				</div>
			</div>
		  <div class="row products_container">

				<!-- Promo Item -->
				<div class="col-lg-4 promo_col">
					<div class="promo_item">
						<div class="promo_image">
							<center><img src="images/karnataka/Rama_SET.281113356_std.jpg" alt="" height="220"></center>
							<div class="promo_content promo_content_2">
								<div class="promo_title">kINETICS oF</div>
								<div class="promo_subtitle">KERELA</div>
							</div>
						</div>
						<div class="promo_link"><a href="paintingsofkerela.html">Shop Now</a></div>
					</div>
				</div>
				<!-- Promo Item -->
				<div class="col-lg-4 promo_col">
					<div class="promo_item">
						<div class="promo_image">
							<center><img src="images/rajasthan/Desert-Dawn-watercolor-painting-by-ananta-mandal_0.jpg" alt="" height="220"></center>
							<div class="promo_content promo_content_2">
								<div class="promo_title">rAREST oF</div>
								<div class="promo_subtitle">rAJASTHANI</div>
							</div>
						</div>
						<div class="promo_link"><a href="paintingsofrajasthan.html">Shop Now</a></div>
					</div>
				</div>
				<!-- Promo Item -->
				<div class="col-lg-4 promo_col">
					<div class="promo_item">
						<div class="promo_image">
							<center><img src="images/nagaland/IMG_20171120_184608_132.jpg" alt="" height="220"></center>
							<div class="promo_content promo_content_2">
								<div class="promo_title">nOSTALGIA oF</div>
								<div class="promo_subtitle">nAGALAND</div>
							</div>
						</div>
						<div class="promo_link"><a href="paintingsofnagaland.html">Shop Now</a></div>
					</div>
				</div>
			</div>
			<div class="row con">
				<!-- Promo Item -->
				<div class="promo_col col-lg-5 offset-lg-1">
					<div class="promo_item">
						<div class="promo_image">
							<center><img src="images/gujarat/Kutch-embroidery-work3.jpg" alt="" height="220"></center>
							<div class="promo_content promo_content_2">
								<div class="promo_title">gLORY oF</div>
								<div class="promo_subtitle">gUJARAT</div>
							</div>
						</div>
						<div class="promo_link"><a href="paintingsofgujarat.html">Shop Now</a></div>
					</div>
				</div>			

				<!-- Promo Item -->
				<div class="col-lg-4 promo_col offset-lg-1">
					<div class="promo_item">
						<div class="promo_image">
							<center><img src="images/kashmir/il_794xN.1133824353_bgft.jpg" alt="" height="220"></center>
							<div class="promo_content promo_content_2">
								<div class="promo_title">kINGS oF </div>
								<div class="promo_subtitle">kASHMIRI</div>
							</div>
						</div>
						<div class="promo_link"><a href="paintingsofjammu.html">Shop Now</a></div>
					</div>
				</div>
			 </div>
		</div>
	</div>
	<!-- Gallery -->

	<div class="gallery">
		<div class="gallery_image" style="background-image:url(images/gallery.jpg)"></div>
		<div class="container">
			<div class="row">
				<div class="col">
					<div class="gallery_title text-center">
						<ul>
							<li><a href="#"></a></li>
							<li><a href="#">ARTISTERY</a></li>
							<li><a href="#"></a></li>
						</ul>
					</div>
					<div class="gallery_text text-center">The Power Of ARTs</div>
					<div class="button gallery_button"><a href="#">The Upcomings</a></div>
				</div>
			</div>
		</div>	
		<div class="gallery_slider_container">
			
			<!-- Gallery Slider -->
			<div class="owl-carousel owl-theme gallery_slider">
				
				<!-- Gallery Item -->
				<div class="owl-item gallery_item">
					<a class="colorbox" href="images/rajasthan/0631.jpg">
						<img src="images/rajasthan/0631.jpg" alt="" height="180">
					</a>
				</div>

				<!-- Gallery Item -->
				<div class="owl-item gallery_item">
					<a class="colorbox" href="images/rajasthan/homedecor.jpg">
						<img src="images/rajasthan/homedecor.jpg" alt="" height="180">
					</a>
				</div>

				<!-- Gallery Item -->
				<div class="owl-item gallery_item">
					<a class="colorbox" href="images/gujarat/GH-8-726x405.jpg">
						<img src="images/gujarat/GH-8-726x405.jpg" alt="" height="180">
					</a>
				</div>

				<!-- Gallery Item -->
				<div class="owl-item gallery_item">
					<a class="colorbox" href="images/gujarat/gujarat.jpg">
						<img src="images/gujarat/gujarat.jpg" alt="" height="180">
					</a>
				</div>

				<!-- Gallery Item -->
				<div class="owl-item gallery_item">
					<a class="colorbox" href="images/karnataka/maxresdefault.jpg">
						<img src="images/karnataka/maxresdefault.jpg" alt="" height="180">
					</a>
				</div>

				<!-- Gallery Item -->
				<div class="owl-item gallery_item">
					<a class="colorbox" href="images/nagaland/713VE-aXS-L._UY625_.jpg">
						<img src="images/nagaland/713VE-aXS-L._UY625_.jpg" alt="" height="180">
					</a>
				</div>

			</div>
		</div>	
	</div>

	<!-- Testimonials -->

	<div class="testimonials">
		<div class="container">
			<div class="row">
				<div class="col">
					<div class="section_title_container text-center">
						<div class="section_subtitle">only the best</div>
						<div class="section_title">testimonials</div>
					</div>
				</div>
			</div>
			<div class="row test_slider_container">
				<div class="col">

					<!-- Testimonials Slider -->
					<div class="owl-carousel owl-theme test_slider text-center">

						<!-- Testimonial Item -->
						<div class="owl-item">
							<div class="test_text">“You will find here a Good Experience for Mobile Shoppers,Easy Checkout Process also a very Easy Checkout Process with a detailed description.”</div>
							<div class="test_content">
								<div class="test_image"><img src="images/wallpaper/animesh.jpg" alt=""></div>
								<div class="test_name">Animesh Mathur</div>
								<div class="test_title">Designer</div>
							</div>
						</div>

						<!-- Testimonial Item -->
						<div class="owl-item">
							<div class="test_text">“You will be a very satisfied customer with assured quality products also Clear Design & Intuitive Navigation.You won't get any change to complain.Fully tested website. ”</div>
							<div class="test_content">
								<div class="test_image"><img src="images/wallpaper/FUDAKTI1.jpg" alt=""></div>
								<div class="test_name">Raghav Sharma</div>
								<div class="test_title">Senior Developer</div>
							</div>
						</div>
						<!-- Testimonial Item -->
						<div class="owl-item">
							<div class="test_text">“You will be a very satisfied customer with assured quality products also Clear Design & Intuitive Navigation.You won't get any change to complain.Fully tested website.”</div>
							<div class="test_content">
								<div class="test_image"><img src="images/Capture1.png" alt=""></div>
								<div class="test_name">Sonakshi Sikhwal</div>
								<div class="test_title">Database Manager</div>
							</div>
						</div>
						<!-- Testimonial Item -->
						<div class="owl-item">
							<div class="test_text">“You will find here a Good Experience for Mobile Shoppers,Easy Checkout Process also a very Easy Checkout Process with a detailed description.You will be a very satisfied customer with assured quality products also Clear Design & Intuitive Navigation.”</div>
							<div class="test_content">
								<div class="test_image"><img src="images/Capture2.png" alt=""></div>
								<div class="test_name">Anirudhi Thanvi</div>
								<div class="test_title">User</div>
							</div>
						</div>
						<!-- Testimonial Item -->
						<div class="owl-item">
							<div class="test_text">“You will find here a Good Experience for Mobile Shoppers,Easy Checkout Process also a very Easy Checkout Process with a detailed description.”</div>
							<div class="test_content">
								<div class="test_image"><img src="images/Capture3.png" alt=""></div>
								<div class="test_name">Tanu Sawlani</div>
								<div class="test_title">Ass. Designer</div>
							</div>
						</div>


						<!-- Testimonial Item -->
						<div class="owl-item">
							<div class="test_text">“You will find here a Good Experience for Mobile Shoppers,Easy Checkout Process also a very Easy Checkout Process with a detailed description.You will be a very satisfied customer with assured quality products also Clear Design & Intuitive Navigation.”</div>
							<div class="test_content">
								<div class="test_image"><img src="images/wallpaper/rishav.jpeg" alt=""></div>
								<div class="test_name">Rishav Sharma</div>
								<div class="test_title">Supervisor and Ass. Developer</div>
							</div>
						</div>

					</div>

				</div>
			</div>
		</div>
	</div>


	<!-- Footer -->

	<footer class="footer">
		<div class="container">
			<div class="row">
				<div class="col text-center">
					<div class="footer_logo"><a href="#">Artistry</a></div>
					<nav class="footer_nav">
						<ul>
							<li><a href="index.html">Home</a></li>
							<li><a href="categories.html">Handicraft</a></li>
							<li><a href="categories.html">Jewellery</a></li>
							<li><a href="categories.html">Paintings</a></li>
							<li><a href="contact.html">contact</a></li>
						</ul>
					</nav>
					<div class="footer_social">
						<ul>
							<li><a href="#"><i class="fa fa-pinterest" aria-hidden="true"></i></a></li>
							<li><a href="#"><i class="fa fa-linkedin" aria-hidden="true"></i></a></li>
							<li><a href="#"><i class="fa fa-instagram" aria-hidden="true"></i></a></li>
							<li><a href="#"><i class="fa fa-reddit-alien" aria-hidden="true"></i></a></li>
							<li><a href="#"><i class="fa fa-twitter" aria-hidden="true"></i></a></li>
						</ul>
					</div>
				</div>
			</div>
		</div>
	</footer>
</div>
<hr color="#DB7115"><hr color="#DB7115">
<script src="js/jquery-3.2.1.min.js"></script>
<script src="styles/bootstrap4/popper.js"></script>
<script src="styles/bootstrap4/bootstrap.min.js"></script>
<script src="plugins/OwlCarousel2-2.2.1/owl.carousel.js"></script>
	<script src="plugins/easing/easing.js"></script>
<script src="plugins/parallax-js-master/parallax.min.js"></script>
<script src="plugins/colorbox/jquery.colorbox-min.js"></script>
<script src="js/custom.js"></script>
<?php
FUNCTION CLICK1()
{
$A="Handicraft";
$B=300;
$C=1;
$D=$C*$B;
$E=RAND(1,10000);
$servername="localhost";
$user="root";
$psw="";
$dbname="cart";
$conn=mysqli_connect($servername,$user,$psw,$dbname);
if(!$conn)
{
die("connection failed:".mysqli_connect_error());
echo "TRy again";
}
$sql="insert into shop values('$A',$B,$C,$D,$E)";
$result=mysqli_query($conn,$sql);
$r="select * from shop";
$u=mysqli_query($conn,$r);
$conn->close();
}
IF(ISSET($_POST["box1"]))
{
CLICK1();
}

FUNCTION CLICK2()
{
$A="Home Decor";
$B=550;
$C=1;
$D=$C*$B;
$E=RAND(1,10000);
$servername="localhost";
$user="root";
$psw="";
$dbname="cart";
$conn=mysqli_connect($servername,$user,$psw,$dbname);
if(!$conn)
{
die("connection failed:".mysqli_connect_error());
echo "TRy again";
}
$sql="insert into shop values('$A',$B,$C,$D,$E)";
$result=mysqli_query($conn,$sql);
$r="select * from shop";
$u=mysqli_query($conn,$r);
$conn->close();
}
IF(ISSET($_POST["box2"]))
{
CLICK2();
}

FUNCTION CLICK3()
{
$A="Sculpture";
$B=990;
$C=1;
$D=$C*$B;
$E=RAND(1,10000);
$servername="localhost";
$user="root";
$psw="";
$dbname="cart";
$conn=mysqli_connect($servername,$user,$psw,$dbname);
if(!$conn)
{
die("connection failed:".mysqli_connect_error());
echo "TRy again";
}
$sql="insert into shop values('$A',$B,$C,$D,$E)";
$result=mysqli_query($conn,$sql);
$r="select * from shop";
$u=mysqli_query($conn,$r);
$conn->close();
}
IF(ISSET($_POST["box3"]))
{
CLICK3();
}

?>
</body>
</html>